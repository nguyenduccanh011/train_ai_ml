import json
d = json.load(open('visualization/data/index.json'))
new_symbols = []
for s in d['stats']:
    sym = s['symbol']
    s['file'] = './data/' + sym + '.json'
    new_symbols.append(s)
d['symbols'] = new_symbols
json.dump(d, open('visualization/data/index.json', 'w'))
print('Fixed. Sample:', json.dumps(new_symbols[0]))
