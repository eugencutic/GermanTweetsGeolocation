import codecs

emb = './data/embedding.txt'
file = codecs.open(emb, 'r', encoding='utf-8')
lines = file.readlines()
file.close()

clean_lines = []
for line in lines:
    word = line.split()[0]
    vector = " ".join(line.split()[1:])
    word = word[2:-1]
    clean_lines.append(" ".join([word, vector]))

clean_emb = './data/clean_embedding.txt'
file = codecs.open(clean_emb, 'w', encoding='utf-8')
for line in clean_lines:
    file.write(line + '\n')
file.close()