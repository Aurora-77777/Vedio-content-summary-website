#!/usr/bin/env python3
"""
PostgreSQL数据库导出脚本（不需要pg_dump）
用于导出RAG向量数据库
"""

import psycopg2
import json
import sys
from pathlib import Path

# 数据库配置（修改为你的配置）
DB_CONFIG = {
    'DB_NAME': 'postgres',
    'DB_USER': 'postgres',
    'DB_PASSWORD': 'mysecretpassword',
    'DB_HOST': 'localhost',
    'DB_PORT': '5433',
    'TABLE_NAME': 'cas_reports'
}

def export_database():
    """导出数据库到JSON文件"""
    try:
        # 连接数据库
        print(f"🔌 正在连接数据库 {DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}...")
        conn = psycopg2.connect(
            database=DB_CONFIG['DB_NAME'],
            user=DB_CONFIG['DB_USER'],
            password=DB_CONFIG['DB_PASSWORD'],
            host=DB_CONFIG['DB_HOST'],
            port=DB_CONFIG['DB_PORT']
        )
        print("✅ 数据库连接成功！")
        
        cur = conn.cursor()
        
        # 获取表结构
        print(f"📋 正在读取表结构...")
        cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{DB_CONFIG['TABLE_NAME']}'
            ORDER BY ordinal_position;
        """)
        columns = cur.fetchall()
        print(f"   找到 {len(columns)} 个字段")
        
        # 导出数据
        print(f"📦 正在导出数据...")
        cur.execute(f"SELECT * FROM {DB_CONFIG['TABLE_NAME']};")
        rows = cur.fetchall()
        print(f"   找到 {len(rows)} 条记录")
        
        # 转换为字典列表
        column_names = [col[0] for col in columns]
        data = []
        for row in rows:
            row_dict = {}
            for i, col_name in enumerate(column_names):
                value = row[i]
                # 处理特殊类型（如vector类型）
                if value is not None and hasattr(value, '__str__'):
                    row_dict[col_name] = str(value)
                else:
                    row_dict[col_name] = value
            data.append(row_dict)
        
        # 保存为JSON
        output_file = 'cas_reports_backup.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'table_name': DB_CONFIG['TABLE_NAME'],
                'columns': [{'name': col[0], 'type': col[1]} for col in columns],
                'data': data
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 数据已导出到: {output_file}")
        print(f"   共 {len(data)} 条记录")
        
        cur.close()
        conn.close()
        
        return output_file
        
    except psycopg2.OperationalError as e:
        print(f"❌ 数据库连接失败: {e}")
        print("\n💡 请检查：")
        print(f"   1. PostgreSQL是否运行在 {DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}？")
        print(f"   2. 用户名 '{DB_CONFIG['DB_USER']}' 和密码是否正确？")
        print(f"   3. 数据库 '{DB_CONFIG['DB_NAME']}' 是否存在？")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    print("=" * 50)
    print("PostgreSQL数据库导出工具")
    print("=" * 50)
    print(f"数据库: {DB_CONFIG['DB_NAME']}")
    print(f"表名: {DB_CONFIG['TABLE_NAME']}")
    print(f"主机: {DB_CONFIG['DB_HOST']}:{DB_CONFIG['DB_PORT']}")
    print("=" * 50)
    print()
    
    # 提示用户修改配置
    print("⚠️  请先修改脚本中的 DB_CONFIG 配置！")
    response = input("是否已修改配置？(y/n): ")
    if response.lower() != 'y':
        print("请先修改脚本中的数据库配置，然后重新运行。")
        sys.exit(0)
    
    export_database()
    print("\n✅ 导出完成！")
