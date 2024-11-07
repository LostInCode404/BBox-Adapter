import os
from openai import OpenAI
from groq import Groq
import numpy as np
from utils.util import extract_first_sentences
from utils.loggers import loggers
from tenacity import wait_random_exponential, stop_after_attempt, retry, RetryError


def log_attempt_number(retry_state):
    """return the result of the last call attempt"""
    loggers["error"].info(f"Retrying: attempt #{retry_state.attempt_number}, wait: {retry_state.outcome_timestamp} for {retry_state.outcome.exception()}")

class LLM_API():
    
    def __init__(self, provider="openai", model="gpt-3.5-turbo", query_params=None, api_key=None):
        self.provider = provider.lower()
        if api_key is None:
            raise ValueError("API key must be provided.")
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "groq":
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose 'openai' or 'groq'.")
        self.query_params = query_params
        self.model = model
        self.token_usage = {"input": 0, "output": 0}
        
        
    @retry(wait=wait_random_exponential(min=2, max=8), stop=stop_after_attempt(30), after=log_attempt_number)
    def query(self, 
            prompt, 
            temp=None,
            n=None,
            stop=None,
            max_tokens=None,
        ):
        prompt_chat = [
            {"role": "user", "content": prompt} 
            ]

        params = {
            "model": self.model,
            "messages": prompt_chat,
            "max_tokens": self.query_params['max_tokens'] if max_tokens is None else max_tokens,
            "temperature": self.query_params['temperature'] if temp is None else temp,
            "frequency_penalty": self.query_params['frequency_penalty'],
            "presence_penalty": self.query_params['presence_penalty'],
            "stop": self.query_params['stop'] if stop is None else stop,
        }

        def make_single_call():
            response = self.client.chat.completions.create(**params)
            self.token_usage['input'] += response.usage.prompt_tokens
            self.token_usage['output'] += response.usage.completion_tokens
            contents = [choice.message.content.strip() for choice in response.choices]
            if len(contents) == 0:
                raise RuntimeError(f"no response from model {self.model}")
            return contents

        if self.provider == "groq":
            params["n"] = 1
            if n and n > 1:
                # For Groq, make multiple sequential calls since n>1 isn't supported
                contents = []
                for _ in range(n):
                    single_contents = make_single_call()
                    contents.extend(single_contents)
                return contents
            else:
                # Normal flow for Groq (n=1 or n=None)
                return make_single_call()
        else:
            # For OpenAI, include n parameter
            params["n"] = self.query_params['num_candidates'] if n is None else n
            return make_single_call()


    def get_response(
                self,
                prompt, 
                temp=None,
                n=None, 
                stop=None,
                max_tokens=None,
                extract_first_sentence=True,
            ):
        try: 
            if extract_first_sentence:       
                return extract_first_sentences(self.query(prompt, temp=temp, n=n, stop=stop, max_tokens=max_tokens))
            else: 
                return [a.strip() for a in self.query(prompt, temp=temp, n=n, stop=stop, max_tokens=max_tokens)]
        except RetryError as e:
            return '<SKIP>'
            
            
    def reset_token_usage(self):
        self.token_usage = {"input": 0, "output": 0}
        
        
    def get_token_usage(self):
        return self.token_usage
    



