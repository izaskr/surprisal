import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
import numpy as np

device = "cpu"
model_id = "dbmdz/german-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelWithLMHead.from_pretrained(model_id).to(device)
logsoftmax = nn.LogSoftmax()

text = "Die meisten Nussarten, die wir heute kennen, stammen ursprünglich aus Asien"

# tokenize
encodings = tokenizer(text, return_tensors="pt")
input_ids = encodings.input_ids.to(device)[0]

target_ids = input_ids.clone()


# get model outputs
with torch.no_grad():
  outputs = model(input_ids, labels=target_ids)

logits = outputs.logits

# for each subword get surprisal, retrieve also the most probable token and its surprisal
for i in range(len(input_ids)):
	current_logits = logits[i]
	current_logprob = logsoftmax(current_logits)
	surprisal_actual_token = -1 * current_logprob[input_ids[i]]
	surprisal_mostprobable_token = -1 * current_logprob[np.argmax(current_logprob).item()]
	mostprobable_token = tokenizer.decode(np.argmax(current_logprob).item())
	current_token = tokenizer.decode(input_ids[i].item())
	print("Current token", current_token, "has surprisal ", surprisal_actual_token, "but most probable token", mostprobable_token,  "has surprisal", surprisal_mostprobable_token)
 
"""
Current token Die has surprisal  tensor(12.6577) but most probable token  Stadt has surprisal tensor(4.7433)
Current token  meisten has surprisal  tensor(10.1034) but most probable token  der has surprisal tensor(2.7407)
Current token  Nuss has surprisal  tensor(9.5232) but most probable token schalen has surprisal tensor(2.0125)
Current token arten has surprisal  tensor(5.9091) but most probable token  sind has surprisal tensor(1.4680)
Current token , has surprisal  tensor(11.5436) but most probable token  die has surprisal tensor(1.0366)
Current token  die has surprisal  tensor(4.4060) but most probable token  in has surprisal tensor(1.6489)
Current token  wir has surprisal  tensor(10.2282) but most probable token  kennen has surprisal tensor(1.8305)
Current token  heute has surprisal  tensor(9.0683) but most probable token  kennen has surprisal tensor(0.8847)
Current token  kennen has surprisal  tensor(9.4372) but most probable token , has surprisal tensor(0.1181)
Current token , has surprisal  tensor(10.0496) but most probable token  sind has surprisal tensor(1.0237)
Current token  stammen has surprisal  tensor(11.7562) but most probable token  aus has surprisal tensor(0.3868)
Current token  ursprünglich has surprisal  tensor(8.3733) but most probable token  aus has surprisal tensor(0.0960)
Current token  aus has surprisal  tensor(9.7171) but most probable token  Afrika has surprisal tensor(2.1346)
Current token  Asien has surprisal  tensor(9.8705) but most probable token . has surprisal tensor(0.9972)
"""
