import os

train_dir = 'dataset/train'
test_dir = 'dataset/test'

print('=== DATASET BALANCE CHECK ===')

print('\nTraining Data:')
train_counts = {}
for class_name in sorted(os.listdir(train_dir)):
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        train_counts[class_name] = count
        print(f'  {class_name}: {count} images')

print('\nTest Data:')
test_counts = {}
for class_name in sorted(os.listdir(test_dir)):
    class_path = os.path.join(test_dir, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        test_counts[class_name] = count
        print(f'  {class_name}: {count} images')

print('\n=== SUMMARY ===')
train_min = min(train_counts.values()) if train_counts else 0
train_max = max(train_counts.values()) if train_counts else 0
test_min = min(test_counts.values()) if test_counts else 0
test_max = max(test_counts.values()) if test_counts else 0
print(f'Training set: min={train_min}, max={train_max}, difference={train_max-train_min}')
print(f'Test set: min={test_min}, max={test_max}, difference={test_max-test_min}')

if train_min == train_max and test_min == test_max:
    print('✅ Your dataset is perfectly balanced!')
else:
    print('⚠️  Your dataset is NOT balanced. Consider balancing for best results.') 