sample_gen_args = {
    "max_new_tokens": 50,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "eos_token_id": 2
}
rej_sample_args = {
    "max_new_tokens": 300,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": True,
    "num_return_sequences": 20,
    "eos_token_id": 20,
}