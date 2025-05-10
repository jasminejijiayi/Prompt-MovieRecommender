import json
import pandas as pd
import os
import csv

def convert_jsonl_to_user_excel(jsonl_file, movies_csv_file, output_file=None):
    """
    将val.jsonl文件转换为Excel表格格式，为每个用户创建两个工作表：
    1. 一个工作表显示用户的观看历史（包含电影风格）
    2. 一个工作表显示用户的候选电影和目标电影（包含电影风格）
    
    参数:
    jsonl_file (str): jsonl文件的路径
    movies_csv_file (str): MovieLens数据集CSV文件的路径，包含电影ID、名称和风格
    output_file (str): 输出Excel文件的路径，默认为'result/user_movies.xlsx'
    
    返回:
    str: 生成的Excel文件路径
    """
    if output_file is None:
        output_file = 'result/user_movies.xlsx'
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取MovieLens数据集并创建电影ID和标题到风格的映射
    movie_genres = {}            # ID -> 风格
    movie_titles = {}            # ID -> 标题
    title_first_letter_map = {}  # (ID, 第一个字母) -> 风格
    
    try:
        # 使用原生csv模块读取文件，它对不规范的CSV文件更宽容
        with open(movies_csv_file, 'r', encoding='latin1') as f:
            # 尝试识别分隔符
            sample = f.read(4096)  # 读取前4096个字节来分析
            dialect = csv.Sniffer().sniff(sample)
            f.seek(0)  # 重置文件指针
            
            # 使用可能更优的分隔符
            csv_reader = csv.reader(f, dialect)
            
            # 读取列名
            headers = next(csv_reader)
            print(f"检测到的文件列名: {headers}")
            
            # 确定列索引
            id_idx = 0  # 第一列应该是ID
            title_idx = 1  # 第二列应该是标题
            genres_idx = 2  # 第三列应该是风格
            
            # 处理所有行
            row_count = 0
            for row in csv_reader:
                try:
                    if len(row) >= 3:  # 确保有足够的列
                        movie_id = int(row[id_idx])
                        title = row[title_idx].strip()
                        
                        # 如果有第四列且非空，使用第四列作为风格
                        if len(row) >= 4 and row[3].strip():
                            genres = row[3].strip()
                            print(f"  使用第四列作为风格: {movie_id}, {title}, {genres}")
                        else:
                            genres = row[genres_idx].strip()
                        
                        # 添加到映射
                        movie_genres[movie_id] = genres
                        movie_titles[movie_id] = title
                        
                        # 创建基于ID和电影第一个字母的映射
                        if len(title) > 0:
                            first_letter = title[0].lower()  # 取第一个字母并转小写
                            title_first_letter_map[(movie_id, first_letter)] = genres
                        
                        row_count += 1
                except Exception as e:
                    # 打印异常行的信息但继续处理
                    print(f"  警告: 第{row_count+2}行数据处理错误: {e}, 行内容: {row}")
                    continue
            
            print(f"已成功读取 {row_count} 行电影数据，包含 {len(movie_genres)} 部电影的风格信息")
                
        print(f"已成功读取电影数据集，包含 {len(movie_genres)} 部电影的风格信息")
    except Exception as e:
        print(f"警告: 读取电影数据集时出错: {e}")
        print("将继续执行，但电影风格信息可能不完整。")
        
    # 创建获取电影风格的函数
    def get_movie_genre(movie_id, movie_title):
        # 首先尝试直接使用ID匹配
        if movie_id in movie_genres:
            return movie_genres[movie_id]
        
        # 如果无法用ID匹配，尝试使用ID和电影标题第一个字母匹配
        if movie_title and len(movie_title) > 0:
            first_letter = movie_title[0].lower()
            if (movie_id, first_letter) in title_first_letter_map:
                return title_first_letter_map[(movie_id, first_letter)]
        
        # 如果都无法匹配，返回未知
        return "未知"
    
    # 读取jsonl文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 创建Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # 为每个用户创建两个工作表
        for entry in data:
            user_id = entry['user_id']
            target_id, target_title = entry['target_item']
            
            # 创建用户观看历史的DataFrame
            history_data = []
            for i, (movie_id, movie_title) in enumerate(entry['item_list'], 1):
                # 获取电影风格信息（尝试使用ID和电影标题第一个字母匹配）
                genres = get_movie_genre(movie_id, movie_title)
                
                history_data.append({
                    '序号': i,
                    '电影ID': movie_id,
                    '电影名称': movie_title,
                    '电影风格': genres
                })
            
            history_df = pd.DataFrame(history_data)
            
            # 创建用户候选电影的DataFrame
            candidate_data = []
            for i, (movie_id, movie_title) in enumerate(entry['candidates'], 1):
                is_target = "是" if movie_id == target_id else "否"
                # 获取电影风格信息（尝试使用ID和电影标题第一个字母匹配）
                genres = get_movie_genre(movie_id, movie_title)
                
                candidate_data.append({
                    '序号': i,
                    '电影ID': movie_id,
                    '电影名称': movie_title,
                    '电影风格': genres,
                    '是否为目标电影': is_target
                })
            
            candidate_df = pd.DataFrame(candidate_data)
            
            # 将DataFrame写入Excel工作表
            history_sheet_name = f"用户{user_id}-观看历史"
            candidate_sheet_name = f"用户{user_id}-候选电影"
            
            history_df.to_excel(writer, sheet_name=history_sheet_name, index=False)
            candidate_df.to_excel(writer, sheet_name=candidate_sheet_name, index=False)
            
            # 格式化工作表
            for sheet_name in [history_sheet_name, candidate_sheet_name]:
                worksheet = writer.sheets[sheet_name]
                # 设置列宽
                worksheet.column_dimensions['A'].width = 10
                worksheet.column_dimensions['B'].width = 15
                worksheet.column_dimensions['C'].width = 40
                worksheet.column_dimensions['D'].width = 30
                if sheet_name == candidate_sheet_name:
                    worksheet.column_dimensions['E'].width = 20
                
                # 添加标题信息
                if sheet_name == history_sheet_name:
                    title = f"用户 {user_id} 的观看历史 (共 {len(history_data)} 部电影)"
                    worksheet.cell(row=1, column=1).value = title
                    worksheet.merge_cells('A1:D1')  # 合并到D列以包含电影风格列
                else:
                    title = f"用户 {user_id} 的候选电影 (目标电影: {target_title}, ID: {target_id})"
                    worksheet.cell(row=1, column=1).value = title
                    worksheet.merge_cells('A1:E1')  # 合并到E列以包含电影风格列和目标电影列
                
                worksheet.insert_rows(2)  # 插入空行作为间隔
    
    print(f"已创建Excel表格: {output_file}")
    print(f"共 {len(data)} 个用户，每个用户2个工作表，总计 {len(data)*2} 个工作表")
    
    return output_file

def create_summary_csv(jsonl_file, output_file=None):
    """
    创建一个简化的CSV摘要文件，每行包含用户ID、目标电影和候选电影数量
    
    参数:
    jsonl_file (str): jsonl文件的路径
    output_file (str): 输出CSV文件的路径，默认为'movie_data_summary.csv'
    
    返回:
    str: 生成的CSV文件路径
    """
    if output_file is None:
        output_file = 'movie_data_summary.csv'
    
    # 读取jsonl文件
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 创建摘要数据
    summary_data = []
    for entry in data:
        user_id = entry['user_id']
        target_id, target_title = entry['target_item']
        
        # 获取观看历史的电影标题，最多显示5部
        watched_titles = [movie[1] for movie in entry['item_list'][:5]]
        watched_titles_str = ', '.join(watched_titles)
        if len(entry['item_list']) > 5:
            watched_titles_str += f"... (共{len(entry['item_list'])}部)"
        
        # 获取候选电影标题，最多显示5部
        candidate_titles = [movie[1] for movie in entry['candidates'][:5]]
        candidate_titles_str = ', '.join(candidate_titles)
        if len(entry['candidates']) > 5:
            candidate_titles_str += f"... (共{len(entry['candidates'])}部)"
        
        summary_data.append({
            'user_id': user_id,
            'target_movie': f"{target_title} (ID: {target_id})",
            'watched_movies': watched_titles_str,
            'candidate_movies': candidate_titles_str
        })
    
    # 创建DataFrame并保存为CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"已创建数据摘要CSV文件: {output_file}")
    return output_file

if __name__ == "__main__":
    # 指定输入文件路径
    jsonl_file = 'val.jsonl'
    movies_csv_file = 'MovieLens 1M dataset _movies.csv'
    output_excel = 'result/user_movies.xlsx'
    
    # 转换为Excel格式（每个用户2个工作表，包含电影风格）
    excel_file = convert_jsonl_to_user_excel(jsonl_file, movies_csv_file, output_excel)
    
    # 创建摘要CSV
    csv_file = create_summary_csv('val.jsonl', 'result/movie_data_summary.csv')
    
    print("\n转换完成!")
    print(f"Excel表格文件: {excel_file}")
    print(f"CSV摘要文件: {csv_file}")
