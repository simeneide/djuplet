import random
import re

def find_valid_punctuation_positions(text, punctuation={',', '.', '?', ':', '!'}):
    """Finds positions where punctuation can be safely added without creating duplicates like ',,' or '!!'."""
    positions = []
    for i in range(1, len(text)):
        if text[i] not in punctuation and text[i-1] not in punctuation:
            positions.append(i)
    return positions

def total_punctuation(text, punctuation={',', '.', '?', ':', '!'}):
    """Returns the total number of punctuation marks in the text."""
    return sum(1 for char in text if char in punctuation)

def corrupt_paragraph(paragraph, level):
    PUNCTUATION = {',', '.', '?', ':', '!'}
    
    if level == 0:
        return paragraph
    
    def move_one_punctuation(text):
        punctuation_positions = [m.start() for m in re.finditer(r'[,.?:!]', text)]
        if not punctuation_positions:
            return text
        
        pos = random.choice(punctuation_positions)
        char = text[pos]
        text = text[:pos] + text[pos+1:]
        valid_positions = find_valid_punctuation_positions(text, PUNCTUATION)
        if valid_positions:
            new_pos = random.choice(valid_positions)
            text = text[:new_pos] + char + text[new_pos:]
        return text

    def move_multiple_punctuation(text):
        total_punct = total_punctuation(text, PUNCTUATION)
        if total_punct <= 1:
            return text
        num = random.randint(1, min(3, total_punct - 1))
        for _ in range(num):
            text = move_one_punctuation(text)
        return text

    def remove_one_punctuation(text):
        punctuation_positions = [m.start() for m in re.finditer(r'[,.?:!]', text)]
        if not punctuation_positions:
            return text
        pos = random.choice(punctuation_positions)
        return text[:pos] + text[pos+1:]

    def remove_multiple_punctuation(text):
        total_punct = total_punctuation(text, PUNCTUATION)
        if total_punct == 0:
            return text
        num = random.randint(1, min(3, total_punct))
        for _ in range(num):
            text = remove_one_punctuation(text)
        return text

    def add_punctuation(text):
        num = random.randint(1, 5)
        valid_positions = find_valid_punctuation_positions(text, PUNCTUATION)
        
        if not valid_positions:
            return text
        
        for _ in range(num):
            if not valid_positions:
                break
            pos = random.choice(valid_positions)
            char = random.choice(list(PUNCTUATION))
            text = text[:pos] + char + text[pos:]
            
            # Adjust valid positions dynamically
            valid_positions = [p + 1 if p >= pos else p for p in valid_positions]
        
        return text

    def randomize_casing(text):
        return ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in text)

    def remove_all_punctuation(text):
        return re.sub(r'[,.?:!]', '', text)

    def lowercase_no_punctuation(text):
        return remove_all_punctuation(text).lower()

    def random_combo(text):
        functions = [move_one_punctuation, move_multiple_punctuation, remove_one_punctuation,
                    remove_multiple_punctuation, add_punctuation, randomize_casing,
                    remove_all_punctuation, lowercase_no_punctuation]
        for _ in range(5):
            f1, f2 = random.sample(functions, 2)
            modified_text = f2(f1(text))
            if modified_text != text:
                return modified_text
        return text

    transformations = {
        1: move_one_punctuation,
        2: move_multiple_punctuation,
        3: remove_one_punctuation,
        4: remove_multiple_punctuation,
        5: add_punctuation,
        6: randomize_casing,
        7: remove_all_punctuation,
        8: lowercase_no_punctuation,
        9: random_combo
    }
    
    return transformations[level](paragraph)

# Example Usage
paragraph = "This is an example paragraph. It has punctuation, and different cases!"
for i in range(10):
    print(f"Level {i}:", corrupt_paragraph(paragraph, i))
