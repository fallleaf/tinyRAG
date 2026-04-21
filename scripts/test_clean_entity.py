#!/usr/bin/env python3
"""
测试实体名称清洗函数

验证 clean_entity_name 函数的过滤效果
"""

import sys
sys.path.insert(0, '/home/fallleaf/tinyRAG')

from plugins.tinyrag_memory_graph.extractor import clean_entity_name


def test_clean_entity_name():
    """测试实体名称清洗函数"""
    
    test_cases = [
        # (输入，预期输出，说明)
        ("```tasks", "tasks", "Markdown 代码块"),
        ('"万', None, "引号（长度<2）"),
        (")", None, "纯符号"),
        (") {", None, "多个符号"),
        ("1.0.3", "1.0.3", "版本号（有效）"),
        ("GB", "GB", "技术术语（有效）"),
        ("极简网络", "极简网络", "中文（有效）"),
        ("2XHpn8kL9mQ3vR7tY1wZ", "2XHpn8kL9mQ3vR7tY1wZ", "乱码（长度<50）"),
        ("[[链接]]", "链接", "Markdown 链接"),
        ("  空格  ", "空格", "首尾空格"),
        ("", None, "空字符串"),
        ("a", None, "长度<2"),
        ("123", None, "纯数字（过滤）"),
        ("!!!", None, "纯符号"),
        ("abc123def456", "abc123def456", "字母数字混合（有效）"),
        ("@#$%", None, "特殊字符过多"),
        ("Hello World", "Hello World", "正常文本"),
        ("项目 A", "项目 A", "中文项目名"),
        ("v1.0.0", "v1.0.0", "版本号"),
    ]
    
    passed = 0
    failed = 0
    
    print("=== 测试实体名称清洗函数 ===\n")
    
    for input_name, expected, description in test_cases:
        result = clean_entity_name(input_name)
        
        if result == expected:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"{status} {description}")
        print(f"   输入：{repr(input_name)}")
        print(f"   期望：{repr(expected)}")
        print(f"   实际：{repr(result)}")
        print()
    
    print(f"\n=== 测试结果 ===")
    print(f"通过：{passed}/{len(test_cases)}")
    print(f"失败：{failed}/{len(test_cases)}")
    
    if failed == 0:
        print("\n✅ 所有测试通过！")
        return True
    else:
        print(f"\n❌ 有 {failed} 个测试失败")
        return False


if __name__ == "__main__":
    success = test_clean_entity_name()
    sys.exit(0 if success else 1)
