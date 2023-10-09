#-*- coding: gbk -*-
# �������ı��ļ�������ı��ļ�
with open('./dataset/friendster/friendster.ungraph.txt', 'r') as input_file, open('output.txt', 'w') as output_file:
    lines = input_file.readlines()

    # ���ڴ洢��������ID֮���ӳ��
    num_to_new_id = {}

    # ����ÿһ�в���ȡΨһ������
    unique_nums = set()
    for line in lines:
        nums = line.strip().split()
        unique_nums.update(nums)

    # �������ֵĴ�С����
    sorted_nums = sorted(map(int, unique_nums))

    # Ϊÿ�����ַ����µ�ID
    for new_id, num in enumerate(sorted_nums):
        num_to_new_id[num] = new_id

    # ���±���ÿһ�в�������ӳ��Ϊ��ID���������д������ļ�
    for line in lines:
        nums = line.strip().split()
        mapped_nums = [str(num_to_new_id[int(num)]) for num in nums]
        output_line = ' '.join(mapped_nums) + '\n'
        output_file.write(output_line)

# ��ӡӳ����
print("ӳ����:")
for num, new_id in num_to_new_id.items():
    print(f"���� {num} ӳ��Ϊ��ID {new_id}")

print("����ѱ��浽output.txt�ļ���")

# �򿪵ڶ��������ı��ļ�������ı��ļ�
with open('dataset/friendster/friendster.community.txt', 'r') as second_input_file, open('second_output.txt', 'w') as second_output_file:
    second_lines = second_input_file.readlines()

    # ���±����ڶ����ļ���ÿһ�в�������ӳ��Ϊ��ID���������д��ڶ�������ļ�
    for line in second_lines:
        nums = line.strip().split()
        mapped_nums = [str(num_to_new_id.get(int(num), num)) for num in nums]
        output_line = ' '.join(mapped_nums) + '\n'
        second_output_file.write(output_line)

print("�ڶ����ļ��е��������滻�����浽second_output.txt�ļ��С�")
