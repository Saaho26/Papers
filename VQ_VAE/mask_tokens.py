import random

def mask_tokens(sentence , mask_token = "[MASK]"):
        words = sentence.split()
        mask_prob = 0.25
        masked_sentence = []
        for word in words:
          if random.random() > mask_prob:
                masked_sentence.append(mask_token)
          else:
                masked_sentence.append(word)


