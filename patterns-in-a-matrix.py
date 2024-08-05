import numpy as np

def find_patterns(matrix):
    from collections import defaultdict, Counter

    def get_patterns_from_line(line):
        patterns = defaultdict(int)
        for length in range(2, len(line) + 1):
            for start in range(len(line) - length + 1):
                pattern = tuple(line[start:start + length])
                patterns[pattern] += 1
        return patterns

    def filter_subpatterns(patterns):
        filtered_patterns = {}
        for pattern in sorted(patterns, key=len, reverse=True):
            if patterns[pattern] > 1:
                is_subpattern = False
                for other_pattern in filtered_patterns:
                    if pattern != other_pattern and len(pattern) < len(other_pattern):
                        if any(pattern == other_pattern[i:i+len(pattern)] for i in range(len(other_pattern) - len(pattern) + 1)):
                            is_subpattern = True
                            break
                if not is_subpattern:
                    filtered_patterns[pattern] = patterns[pattern]
        return filtered_patterns

    
    row_patterns = Counter()
    col_patterns = Counter()

    for row in matrix:
        row_patterns.update(get_patterns_from_line(row))
    
    for col in zip(*matrix):
        col_patterns.update(get_patterns_from_line(col))
    
    
    all_patterns = row_patterns + col_patterns

    
    filtered_patterns = filter_subpatterns(all_patterns)

    return filtered_patterns

r = int(input("enter no of rows:"))
c = int(input("enter no of columns:"))
matrix = np.ndarray((r,c),dtype=int)

print("enter the matrix")
for i in range(0,r):
    for j in range(0,c):
        matrix[i,j] = int(input())

patterns = find_patterns(matrix)
for pattern, count in patterns.items():
    print(f"{' '.join(map(str, pattern))} occurs {count} times")