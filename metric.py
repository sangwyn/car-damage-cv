from torchvision.io import decode_image
import torch

parts = ['Trunk', 'Back-windshield', 'Mirror_Headlight', 'None', 'Rocker-panel_Back-wheel', 'Grille', 'Back-door', 'Front-wheel',
         'Windshield_Fender', 'Tail-light_Quarter-panel', 'License-plate_Front-bumper', 'Hood', 'Front-door']

dmgs = ['Missing part', 'Cracked', 'Scratch', 'Broken part', 'None', 'Corrosion', 'Dent', 'Flaking', 'Paint chip']

part_weights = {
    "Windshield_Fender": 10,
    "Back-windshield": 9,   # Structural/safety importance
    "Tail-light_Quarter-panel": 8,
    "Front-wheel": 8,       # Critical for movement
    "Rocker-panel_Back-wheel": 8,
    "Mirror_Headlight": 8,
    "Hood": 7,              # Engine protection
    "License-plate_Front-bumper": 6,
    "Front-door": 6,        # Safety/function
    "Back-door": 6,
    "Trunk": 6,             # Storage/water ingress risk
    "Grille": 4,            # Cosmetic/minor function
    "None": 0
}

dmg_weights = {
    "Missing part": 10,   # Complete failure
    "Broken part": 9,     # Severe damage
    "Cracked": 8,         # Structural compromise
    "Dent": 7,            # Affects structure/aerodynamics
    "Flaking": 6,         # Leads to corrosion
    "Corrosion": 5,       # Long-term degradation
    "Paint chip": 4,      # Cosmetic/minor
    "Scratch": 3,          # Superficial
    "None": 0
}

res = [[6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                      [4, 3, 0, 2, 0, 2, 2, 0, 4, 4, 4, 5, 4],
                      [1, 0, 0, 2, 0, 1, 1, 0, 2, 1, 1, 3, 1],
                      [2, 3, 0, 2, 1, 2, 2, 0, 2, 2, 2, 3, 3],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                      [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                      [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
for i, dmg in enumerate(dmgs):
    for j, part in enumerate(parts):
        res[i][j] = dmg_weights[dmg] * part_weights[part]
print(res)

def get_metric(part_mask: torch.Tensor, damage_mask: torch.Tensor):
    height, width = part_mask.shape[1], part_mask.shape[2]
    num_parts, num_damages = part_mask.shape[0], damage_mask.shape[0]

    # Initialize
    total_score = 0.0
    x_min, y_min = width, height  # Initialize to maximum possible values
    x_max, y_max = 0, 0
    part_pixels = {part: {dmg: 0 for dmg in dmgs} for part in parts}
    part_pixels = {part: {**part_pixels[part], 'total': 0} for part in parts}

    # Vectorize the pixel iteration using torch.nonzero and boolean indexing
    part_indices = torch.nonzero(part_mask)  # (N, 3), N is the number of non-zero pixels
    damage_indices = torch.nonzero(damage_mask)
    
    # Optimized version
    for k, i, j in part_indices:
        part = parts[k]

        # Update bounding box
        x_min = min(x_min, j.item())
        y_min = min(y_min, i.item())
        x_max = max(x_max, j.item())
        y_max = max(y_max, i.item())

        part_pixels[part]['total'] += 1

        for l_idx, dmg_idx, dmg_jdx in damage_indices:
            if i == dmg_idx and j == dmg_jdx:
                dmg = dmgs[l_idx]
                part_pixels[part][dmg] += 1

    # Calculate total score
    for part in parts:
        part_total = part_pixels[part]['total']
        if part_total > 0:  # avoid division by zero
            for dmg in dmgs:
                dmg_area = part_pixels[part][dmg] / part_total
                total_score += part_weights[part] * dmg_weights[dmg] * dmg_area

    # Calculate car area
    car_area = (x_max - x_min + 1) * (y_max - y_min + 1)

    # Calculate normalized score and grade
    max_possible_weight = max(max(part_weights.values()), max(dmg_weights.values()))
    normalized_score = (total_score / (car_area * max_possible_weight)) * 100 if car_area > 0 else 0
    grade = int(normalized_score // 20) + 1

    return normalized_score, grade


def get_grade(img_path, model_parts = None, model_damage = None):
    img_tensor = decode_image(img_path)
    # img_tensor = torch.frombuffer(img_path, dtype=torch.uint8)

    return img_tensor, img_tensor, img_tensor, 1, 1

    if model_parts is None:
        model_parts = torch.load('./model_parts.pth')
    if model_damage is None:
        model_damage = torch.load('./model_damage.pth')

    mask_parts = model_parts(img_tensor)
    mask_damage = model_damage(img_tensor)
    normalized_score, grade = get_metric(mask_parts, mask_damage)

    return img_tensor, mask_parts, mask_damage, normalized_score, grade


def get_descr(img, part_mask, damage_mask, x, y):
    found = {'parts': [], 'dmgs': []}
    i, j = x, y
    for k, part in enumerate(parts):
        if part_mask[k, i, j] == 0:
            continue

        x_min, y_min = min(x_min, j), min(y_min, i)
        x_max, y_max = max(x_max, j), max(y_max, i)

        if part not in found['parts']:
            found['parts'].append(part)

        for l, dmg in enumerate(dmgs):
            if damage_mask[l, i, j] == 0:
                continue
            
            if dmg not in found['dmgs']:
                found['dmgs'].append(dmg)

    return f'Defects: {found['dmgs']} | Parts: {found['parts']}'
