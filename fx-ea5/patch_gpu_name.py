#!/usr/bin/env python3
import sys

f = open('f:/FX/ai_ea/run_train.py', 'r', encoding='utf-8')
src = f.read()
f.close()

if 'GPU_NAME' in src:
    print('GPU_NAME already exists - skipping')
    sys.exit(0)

# Insert GPU_NAME after the NODE_ID = _detect_node_id() line
lines = src.split('\n')
new_lines = []
inserted = False
for line in lines:
    new_lines.append(line)
    if not inserted and line.strip().startswith('NODE_ID') and '_detect_node_id' in line:
        gn = 'GPU_NAME = os.environ.get(' + "'GPU_NAME'" + ', NODE_ID.upper())  # ダッシュボード表示用'
        new_lines.append(gn)
        inserted = True

src = '\n'.join(new_lines)
with open('f:/FX/ai_ea/run_train.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('GPU_NAME inserted:', inserted)
