#!/usr/bin/env python3
"""Add gpu_name/node_id to progress dict in run_train.py"""

f = open('f:/FX/ai_ea/run_train.py', 'r', encoding='utf-8')
src = f.read()
f.close()

key = "'vram_total_gb':"
idx = src.find(key)
if idx < 0:
    print('vram_total_gb not found')
    raise SystemExit(1)

# Find the next occurrence of 'message': after vram_total_gb
after = src[idx:]
msg_idx = after.find("'message':")
if msg_idx < 0:
    print('message not found after vram_total_gb')
    raise SystemExit(1)

insert_pos = idx + msg_idx
before = src[:insert_pos]
after2 = src[insert_pos:]

# Check if already patched
if "'gpu_name':" in before[idx:insert_pos] or "'node_id':" in before[idx:insert_pos]:
    print('Already patched - skipping')
    raise SystemExit(0)

# Find the line ending just before 'message':
# We want to insert after the 'vram_total_gb' line
# Find the \n before 'message':
rn = before.rfind('\n')
insert_after_newline = rn + 1
indent = ''
for ch in after2:
    if ch in (' ', '\t'):
        indent += ch
    else:
        break

new_lines = (
    f"{indent}'gpu_name':        GPU_NAME,\n"
    f"{indent}'node_id':         NODE_ID,\n"
)

src = src[:insert_pos] + new_lines + src[insert_pos:]

with open('f:/FX/ai_ea/run_train.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('Patched successfully')

# verify
src2 = open('f:/FX/ai_ea/run_train.py', encoding='utf-8').read()
idx2 = src2.find("'gpu_name':")
if idx2 > 0:
    print('Verification OK:', src2[idx2:idx2+50])
else:
    print('Verification FAILED - gpu_name not found')
