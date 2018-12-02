from PIL import Image
import sys
import os

# example run: python resize.py 32 data/train/ data/train32/

def main():
  if len(sys.argv) < 4:
    raise Exception('not enough arguments. python resize.py <size> <target dir> <output dir>')

  size = int(sys.argv[1])
  target = sys.argv[2]  
  output = sys.argv[3]

  # each category
  for directory in os.listdir(target):
    counter = 0

    output_path = os.path.join(output, directory)
    target_path = os.path.join(target, directory)

    # create path
    if not os.path.exists(output_path):
      os.makedirs(output_path)

    # each file per category
    for filename in os.listdir(target_path):
      with Image.open(os.path.join(target_path, filename)) as img:
        resized = img.resize((size, size), Image.ANTIALIAS)
        resized.save(os.path.join(output_path, str(counter) + '.jpg'))
        counter += 1
    
    # reset counter
    counter = 0

if __name__ == '__main__':
  main()