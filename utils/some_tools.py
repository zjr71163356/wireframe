import os
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

def visualize_pointlines(pickle_path):
    """
    读取pickle文件并可视化pointlines，给线段和点标上序号
    
    Args:
        pickle_path (str): pickle文件路径
    """
    # 加载pickle数据
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # 提取需要的数据
    points = data['points']
    pointlines = data['pointlines']
    img = data['img']
    
    # 创建图像
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 如果img是numpy数组，显示图像
    if isinstance(img, np.ndarray):
        ax.imshow(img)
    else:
        # 设置纯白背景
        ax.set_facecolor('white')
    
    # 绘制所有点
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    ax.scatter(x_coords, y_coords, color='blue', s=30, alpha=0.7)
    
    # 标记点的序号
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.annotate(f"{i}", (x, y), fontsize=8, ha='right', va='bottom', 
                   bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
    
    # 绘制pointlines
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for point_idx, point_data in enumerate(pointlines):
        if not point_data or len(point_data) < 3:
            continue
            
        point_coord = point_data[0]  # 点的坐标
        connected_points = point_data[1]  # 连接的点对索引
        
        # 绘制从当前点到其他点的线
        for line_idx, point_pair in enumerate(connected_points):
            p1_idx, p2_idx = point_pair
            
            # 获取两个点的坐标
            if p1_idx < len(points) and p2_idx < len(points):
                p1 = points[p1_idx]
                p2 = points[p2_idx]
                
                # 绘制线段
                color_idx = line_idx % len(colors)
                line = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                              color=colors[color_idx], linewidth=1.5, alpha=0.8)[0]
                
                # 在线段中间标注线段序号
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                
                # 计算线段编号 (可以根据point_idx和line_idx生成唯一标识)
                line_number = f"{p1_idx}-{p2_idx}"
                
                # 标记线段序号
                ax.annotate(line_number, (mid_x, mid_y), fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    
    # 设置图像边界，稍微扩大一点以便显示完整
    plt.xlim(min(x_coords) - 10, max(x_coords) + 10)
    plt.ylim(min(y_coords) - 10, max(y_coords) + 10)
    
    # 添加图像信息
    plt.title(f"Pointlines Visualization\nFile: {os.path.basename(pickle_path)}")
    plt.tight_layout()
    
    # 保存图像
    output_path = os.path.splitext(pickle_path)[0] + '_pointlines_visualization.png'
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # 显示图像
    plt.show()

 

def load_pickle(pickle_path):
    import pickle
    import json
    
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
        # After loading the pickle data
        # Create a text file with same name but .txt extension
        txt_path = os.path.splitext(pickle_path)[0] + '.json'
        # Write pickle data to text file
        with open(txt_path, 'w') as txt_file:
            try:
                # Try JSON format for better readability
                json.dump(pickle_data, txt_file, indent=2, default=str)
            except (TypeError, OverflowError):
                # Fallback to string representation
                txt_file.write(str(pickle_data))
        print(f"Data written to {txt_path}")
        
        return pickle_data
    
# ...existing code...

def json_to_pkl(json_path, output_dir=None, img_dir=None):
    """
    将特定格式的JSON文件转换为PKL格式
    
    Args:
        json_path (str): JSON文件路径
        output_dir (str, optional): 输出目录路径，默认为JSON文件所在目录
        img_dir (str, optional): 图像所在目录，默认为JSON文件所在目录
    
    Returns:
        str: 生成的PKL文件路径
    """
    import json
    import os
    import pickle
    import numpy as np
    import cv2
    import math
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定图像目录
    if img_dir is None:
        img_dir = os.path.dirname(json_path)
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 获取图像名称和路径
    image_name = data.get('imagePath', '')
    image_path = os.path.join(img_dir, image_name)
    
    # 读取图像数据
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    else:
        # 如果找不到图像文件，尝试从JSON的imageData字段解码
        if 'imageData' in data and data['imageData']:
            import base64
            img_data = base64.b64decode(data['imageData'])
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = None
            print(f"警告: 无法找到图像 {image_name}")
    
    # 提取多边形点
    points = []
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'polygon':
            for point in shape.get('points', []):
                if len(point) == 2:
                    points.append(point)
    
    # 构建线集合（每条线由其两个端点的索引表示）
    lines = []
    for i in range(len(points)):
        lines.append([i, (i + 1) % len(points)])
    
    # 构建pointlines（每个点关联的线集合）
    pointlines = [ 
    [points[0],[0,1],[0,3]],
    [points[1],[1,2],[1,0]],
    [points[2],[2,3],[2,1]],
    [points[3],[3,0],[3,2]]
    ]
 
    
    # 构建pointlines_index
    pointlines_index = [[lines.index(l1),lines.index(l2)] for (p,l1,l2) in pointlines]
    
    # 计算junction（交点位置，即points）
    junction = points.copy()
    
    # 计算theta（每个交点分支的角度值）
    theta = []
    for i, point in enumerate(points):
        angles = []
        for line_idx in pointlines[i]:
            p1, p2 = lines[line_idx]
            other_point_idx = p2 if p1 == i else p1
            other_point = points[other_point_idx]
            
            # 计算角度
            dx = other_point[0] - point[0]
            dy = other_point[1] - point[1]
            angle = math.atan2(dy, dx)
            angles.append(angle)
        theta.append(angles)
    
    # 构建输出数据
    output_data = {
        'imagename': image_name,
        'img': img,
        'points': np.array(points, dtype=np.float32),
        'lines': np.array(lines, dtype=np.int32),
        'pointlines': pointlines,
        'pointlines_index': np.array(pointlines_index, dtype=np.int32),
        'junction': np.array(junction, dtype=np.float32),
        'theta': theta
    }
    
    # 保存为PKL文件
    pkl_filename = os.path.splitext(os.path.basename(json_path))[0] + '.pkl'
    pkl_path = os.path.join(output_dir, pkl_filename)
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"已生成PKL文件: {pkl_path}")
    return pkl_path

# ...existing code...
def get_filelist(dir='test',data_root='data'):
    # 定义测试目录路径
    # 如果test目录不在当前工作目录，请提供绝对路径，例如 '/path/to/test'

    # 定义想要包含的图片文件扩展名（可以根据需要添加更多）
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'}

    # 输出的文本文件名
    output_file = os.path.join(dir+'_new.txt')

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
    # print("this is sometools")
    # get_filelist('/home/tyrfly1001/wireframe/data/v1.1/test')
    import sys
    if len(sys.argv) > 1:
        pickle_path = sys.argv[1]
        load_pickle(pickle_path)
        visualize_pointlines(pickle_path)
    else:
        print("请提供pickle文件路径")
    
    