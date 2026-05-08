import sentencepiece as spm

try:
    sp = spm.SentencePieceProcessor()
    sp.load('wikitext_bpe.model')
    print("Testing decode with valid")
    print(sp.decode([0, 1, 2, 3]))
    
    print("Testing decode with out of bounds")
    try:
        sp.decode([8005])
        print("Success 8005")
    except Exception as e:
        print(f"Error 8005: {e}")
        
    print("Testing decode with negative")
    try:
        sp.decode([-1])
        print("Success -1")
    except Exception as e:
        print(f"Error -1: {e}")
        
except Exception as e:
    print(f"Init Error: {e}")
