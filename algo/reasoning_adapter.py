from algo.adapter import Adapter
from utils.util import accumulate_strings
from utils.loggers import loggers
import wandb
from concurrent.futures import TimeoutError
from algo.beam_search import Beam_Search
from copy import copy
from accelerate.utils import gather_object
from tqdm.auto import tqdm
import numpy as np
from utils.robustness_transforms import RobustnessTransforms

class Reasoning_Adapter(Adapter):
    
    def __init__(self, prompt, config):
        self.config = config
        self.prompt = prompt
        self.token_usage = {"input": 0, "output": 0}
        
        self.stop_criterion = None
        self.get_accuracy = None
        self.is_correct = None
        self.qa_template = None

        super().__init__(
                config=config,
                prompt=prompt,
            )
        self.acc_table = wandb.Table(columns=["stage", "accuracy"])
 
 
    def get_ans_from_blackbox(self, q, n=1, temp=1):
        qa_text = self.formulate_qa(q=q, a="")
        prompt = f"{self.prompt}\n{qa_text}"
        ans = self.generator.get_response(
                prompt, 
                n=n, 
                stop=["\n\n"], 
                max_tokens=500, 
                extract_first_sentence=False,
                temp=temp
            )
        loggers["api"].info(f"\n{'='*20}\nQuery:\n{prompt}\nResponses:\n{ans}")
        return ans
        

    def prepare_for_training(self, batch, dataset_path="", use_adapter=True):
        
        def process_batch_item(b):
            ground_truth = self.extract_ground_truth(b)
            try:
                question = self.formulate_question(b)
                if use_adapter:
                    beam_search = Beam_Search(
                                params=self.config,
                                thought_generator=self.thought_generator,
                                init_sequence=question,
                                stop_criterion=self.stop_criterion,
                                qa_template=self.qa_template
                            )
                    negative_ans = beam_search(return_with_init=False)[:self.config["num_negatives_for_training"]]
                else:
                    negative_ans = self.get_ans_from_blackbox(q=question, n=self.config["num_candidates_blackbox_warmup"])
                    
                if negative_ans is None:
                    return None
                
                negative_ans = list(set(negative_ans)) # simple deduplication
                negative_ans = [ans.strip() for ans in negative_ans]
                if self.config["only_eval_answers"]:
                    neg_texts = [ans for ans in accumulate_strings(negative_ans)]
                else:
                    neg_texts = [self.formulate_qa(q=question, a=ans) for ans in accumulate_strings(negative_ans)]
                return neg_texts, ground_truth
            
            except TimeoutError:
                loggers["error"].info("[prepare] Beam search timed out.")
                return None
            except Exception as e:
                loggers["error"].info(f"[prepare] Error in process_batch_item: {e}")
                return None
            
        list_idx = list(range(len(batch)))
        progress_bar = tqdm(total=len(list_idx), desc="prepare", disable=not self.accelerator.is_local_main_process)
        self.accelerator.wait_for_everyone()
        with self.accelerator.split_between_processes(list_idx) as batch_idx:
            results = dict(negative_texts=[], positive_texts=[])
            for idx in batch_idx:
                b = batch[idx]
                result = process_batch_item(b)
                if result:
                    completions, ground_truth = result
                    for c in completions:
                        if self.is_correct(c, ground_truth) and self.config["use_outcome_supervision"]:
                            results["positive_texts"].append(c)
                        else:
                            results["negative_texts"].append(c)
                progress_bar.update(self.accelerator.num_processes)
            results = [results]
            
        gathered_results = gather_object(results)
        if self.accelerator.is_main_process:  
            positive_texts = []
            negative_texts = []
            
            for b in batch:
                question = self.formulate_question(b)
                positive_ans = self.get_positive_ans(b)
                if self.config["only_eval_answers"]:
                    positive_texts.extend([ans for ans in accumulate_strings(positive_ans)])
                else:
                    positive_texts.extend([self.formulate_qa(q=question, a=ans) for ans in accumulate_strings(positive_ans)])

            for result in gathered_results:
                negative_texts.extend(result["negative_texts"])
                positive_texts.extend(result["positive_texts"])
                
            self.build_dataset(
                    positive_texts, 
                    negative_texts, 
                    save_to=dataset_path
                )


    def evaluate(self, eval_dataset, use_adapter=True, stage_name=""):
        
        # Initialize robustness transformer
        robustness_transformer = RobustnessTransforms(self.config)
        
        def process_batch_item(b, temperature=1.0, transform_type="none"):
            ground_truth = self.extract_ground_truth(b)
            try:
                question = self.formulate_question(b)
                
                # Apply robustness transformation if specified
                if transform_type != "none":
                    question, temp = robustness_transformer.apply_transform(question, transform_type, temperature)
                    temperature = temp
                
                if use_adapter:
                    beam_search = Beam_Search(
                        params={**self.config, "temperature": temperature},
                        thought_generator=self.thought_generator,
                        init_sequence=question,
                        stop_criterion=self.stop_criterion,
                        qa_template=self.qa_template
                    )
                    answer = beam_search(return_with_init=False)
                else:
                    answer = self.get_ans_from_blackbox(q=question, n=1, temp=temperature)
                if answer is None:
                    return None
                
                return answer[0], ground_truth
            
            except TimeoutError:
                loggers["error"].info("[eval] Beam search timed out.")
                return None
            
            except Exception as e:
                loggers["error"].info(f"[eval] Error in process_batch_item: {e}")
                return None
        
        split_dict = {
            "list_idx": list(range(len(eval_dataset))) * self.config["num_eval_rounds"],
            "round_idx": [i for i in range(self.config["num_eval_rounds"]) for _ in range(len(eval_dataset))],
            "temperature": [1.0] * (len(eval_dataset) * self.config["num_eval_rounds"]),
            "transform_type": ["none"] * (len(eval_dataset) * self.config["num_eval_rounds"])
        }
        
        # Add robustness evaluation if configured
        if self.config.get("eval_robustness", False):
            robustness_types = self.config.get("robustness_types", ["temperature"])
            temperatures = self.config.get("robustness_temperature", [0.7, 1.0, 1.3])
            robustness_rounds = self.config.get("num_robustness_rounds", 2)
            # Extend split_dict for all robustness tests
            for transform_type in robustness_types:
                if transform_type == "temperature":
                    temps = temperatures
                else:
                    temps = [1.0]  # Other transforms don't use temperature
                    
                for temp in temps:
                    split_dict["list_idx"].extend(list(range(len(eval_dataset))) * robustness_rounds)
                    split_dict["round_idx"].extend([
                        f"robustness_{transform_type}_t{temp}_r{r}" 
                        for r in range(robustness_rounds) 
                        for _ in range(len(eval_dataset))
                    ])
                    split_dict["temperature"].extend([temp] * (robustness_rounds * len(eval_dataset)))
                    split_dict["transform_type"].extend([transform_type] * (robustness_rounds * len(eval_dataset)))
        
        
        progress_bar = tqdm(total=len(split_dict["list_idx"]), desc=stage_name, disable=not self.accelerator.is_local_main_process)
        
        # split the batch between processes and gather the results
        self.accelerator.wait_for_everyone()
        with self.accelerator.split_between_processes(split_dict) as splitted_dict:
            results = dict(completions=[], ground_truths=[], rounds=[], transform_types=[], temperature=[])
            for idx, round, temp, transform_type in zip(
                splitted_dict["list_idx"], 
                splitted_dict["round_idx"],
                splitted_dict.get("temperature", [1.0] * len(splitted_dict["list_idx"])),
                splitted_dict.get("transform_type", ["none"] * len(splitted_dict["list_idx"]))
            ):
                b = eval_dataset[idx]
                result = process_batch_item(b, temperature=temp, transform_type=transform_type)
                if result:
                    completion, ground_truth = result
                    results["completions"].append(completion)
                    results["ground_truths"].append(ground_truth)
                    results["rounds"].append(round)
                    results["transform_types"].append(transform_type)
                    results["temperature"].append(temp)
                progress_bar.update(self.accelerator.num_processes)
            results = [results]
        gathered_results = gather_object(results)
        
        # Calculate eval results
        if self.accelerator.is_main_process:
            results = dict(completions=[], ground_truths=[], rounds=[], transform_types=[], temperature=[])
            for result in gathered_results:
                results["completions"].extend(result["completions"])
                results["ground_truths"].extend(result["ground_truths"])
                results["rounds"].extend(result["rounds"])
                results["transform_types"].extend(result["transform_types"])
                results["temperature"].extend(result["temperature"])
            base_results = {
                "completions": [c for c, r, t in zip(results["completions"], 
                                                    results["rounds"],
                                                    results["transform_types"]) 
                              if t == "none"],
                "ground_truths": [g for g, r, t in zip(results["ground_truths"], 
                                                      results["rounds"],
                                                      results["transform_types"]) 
                                if t == "none"],
                "rounds": [r for r, t in zip(results["rounds"], 
                                           results["transform_types"]) 
                          if t == "none"]
            }
            accuracy, std = self.get_accuracy(base_results)
            print(f"\nStage: {stage_name}, Base Accuracy: {accuracy * 100:.2f}% ± {std * 100:.2f}%")
            data = [stage_name] + [accuracy]
            self.acc_table.add_data(*data)
            self.accelerator.log({"base_accuracy": accuracy})
            self.accelerator.log({"accuracy": copy(self.acc_table)})
            
            # Calculate robustness metrics if enabled
            if self.config.get("eval_robustness", False):
                print("\n=== Robustness Evaluation ===")
                robustness_results = {}
                overall_scores = []
                
                for transform_type in robustness_types:
                    print(f"\nEvaluating {transform_type} robustness:")
                    transform_results = {}
                    transform_accuracies = []
                    if transform_type == "temperature":
                        test_values = temperatures
                    else:
                        test_values = [1.0]
                    for temp in test_values:
                        temp_results = {
                            "completions": [c for c, r, t, rt in zip(results["completions"], results["rounds"], results["transform_types"], results["temperature"]) if t == transform_type and rt == temp],
                            "ground_truths": [g for g, r, t, rt in zip(results["ground_truths"], results["rounds"], results["transform_types"], results["temperature"]) if t == transform_type and rt == temp],
                            "rounds": [r for r, t, rt in zip(results["rounds"], results["transform_types"], results["temperature"]) if t == transform_type and rt == temp]
                        }
                        if (len(temp_results["completions"]) > 0 and 
                            len(temp_results["ground_truths"]) > 0):
                            try:
                                acc, std = self.get_accuracy(temp_results)
                                transform_results[temp] = (acc, std)
                                transform_accuracies.append(acc)
                                print(f"{transform_type} (temp={temp}): {acc * 100:.2f}% ± {std * 100:.2f}%")
                            except Exception as e:
                                print(f"Error calculating accuracy for {transform_type} at temp={temp}: {str(e)}")
                                transform_results[temp] = (0.0, 0.0)
                                transform_accuracies.append(0.0)
                        else:
                            print(f"No valid results for {transform_type} at temp={temp}")
                            transform_results[temp] = (0.0, 0.0)
                            transform_accuracies.append(0.0)
                        
                    # Calculate robustness score for this transform type if we have valid results
                    if transform_accuracies:
                        robustness_score = self.calculate_robustness_score(transform_accuracies, accuracy, transform_type)
                        overall_scores.append(robustness_score)
                        print(f"{transform_type} Robustness Score: {robustness_score:.4f}")
                        print(f"Average Accuracy Drop: {(accuracy - np.mean(transform_accuracies)) * 100:.2f}%")
                        
                        # Log robustness metrics
                        self.accelerator.log({
                            f"robustness_score_{transform_type}": robustness_score,
                            f"accuracy_drop_{transform_type}": accuracy - np.mean(transform_accuracies),
                            f"relative_performance_{transform_type}": np.mean(transform_accuracies) / accuracy if accuracy > 0 else 0
                        })
                    else:
                        print(f"Skipping robustness score calculation for {transform_type} due to no valid results")
                    robustness_results[transform_type] = transform_results
                
                # Calculate overall robustness score only if we have valid scores
                if overall_scores:
                    overall_robustness = np.mean(overall_scores)
                    print(f"\nOverall Robustness Score: {overall_robustness:.4f}")
                    self.accelerator.log({"overall_robustness_score": overall_robustness})
                else:
                    print("\nCould not calculate overall robustness score due to no valid results")


    def calculate_robustness_score(self, accuracies, base_accuracy, transform_type):
        """
        Calculate robustness score based on transform type:
        1. For temperature: consider both consistency and degradation
        2. For others: consider only performance degradation
        """
        if not accuracies:
            return 0.0
            
        # Calculate average performance degradation
        relative_performance = np.mean(accuracies) / base_accuracy if base_accuracy > 0 else 0
        degradation_score = max(0, min(1, relative_performance))  # Bound between 0 and 1
        
        # For temperature, consider both consistency and degradation
        if transform_type == "temperature":
            consistency_score = 1.0 - np.std(accuracies)
            return 0.5 * consistency_score + 0.5 * degradation_score
        # For other transforms, only consider degradation
        else:
            return degradation_score

    def formulate_qa(self, q, a):
        return self.qa_template.replace("<Q>", q).replace("<A>", a)


    def get_positive_ans(self, b):
        pass
    
    
    def formulate_question(self, b):
        pass


    def extract_ground_truth(self, b):
        pass

    

