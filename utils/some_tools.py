import os

def get_filelist(dir='test',data_root='data'):
    # 定义测试目录路径
    # 如果test目录不在当前工作目录，请提供绝对路径，例如 '/path/to/test'

    # 定义想要包含的图片文件扩展名（可以根据需要添加更多）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'}

    # 输出的文本文件名
    output_file = os.path.join(dir+'..', 'test.txt')

    try:
        # 获取test目录中的所有文件
        all_files = os.listdir(dir)
    except FileNotFoundError:
        print(f"目录 '{dir}' 不存在。请检查路径是否正确。")
        exit(1)
    except PermissionError:
        print(f"没有权限访问目录 '{dir}'。")
        exit(1)

    # 过滤出图片文件
    image_files = [file for file in all_files
                  if os.path.isfile(os.path.join(dir, file)) and
                  os.path.splitext(file)[1].lower() in image_extensions]

    # 可选：按文件名排序
    image_files.sort()

    # 将文件名写入文本文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in image_files:
            f.write(f"{file}\n")

    print(f"已将 {len(image_files)} 个图片文件名写入 '{output_file}'。")

if __name__=='__main__':
    print("this is sometools")
    get_filelist('/home/featurize/data/v1.1/test')
