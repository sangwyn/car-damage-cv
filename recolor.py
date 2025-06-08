import json

meta_path = './Car damages dataset/meta.json'

with open(meta_path, 'r') as f:
    meta = json.load(f)

print(meta)

colors = {
'Trunk': '#e74c3c',
'Rocker-panel': '#9b59b6',
'Grille': '#34495e',
'Front-bumper': '#f1c40f',
'Back-wheel': '#9b59b6',
'Front-wheel': '#3498db',
'Back-window': '#f1c40f',
'Front-door': '#e67e22',
'Back-door': '#8e44ad',
'Headlight': '#2c3e50',
'Back-windshield': '#27ae60',
'Quarter-panel': '#2ecc71',
'Rocker-panel': '#9b59b6',
'Windshield': '#16a085',
'Front-window': '#27ae60',
'Hood': '#f39c12',
'Fender': '#16a085',
'Tail-light': '#2ecc71',
'License-plate': '#f1c40f',
'Back-bumper': '#9b59b6',
'Roof': '#f39c12',
'Mirror': '#2c3e50'
}

for i in range(len(meta['classes'])):
    meta['classes'][i]['color'] = colors[meta['classes'][i]['title']].upper()

with open(meta_path, 'w') as f:
    json.dump(meta, f)