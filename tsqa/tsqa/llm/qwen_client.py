from .base import BaseLLMClient
import torch


class QwenClient(BaseLLMClient):

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        image=None,
        artifacts=None,
        **kwargs,
    ) -> str:

        max_new_tokens = kwargs.get("max_tokens", self.max_tokens)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
