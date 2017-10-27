def generate_prefix_file(list_file_path, output_prefix_file_path):

    lines=[]

    with open(list_file_path) as file:
        lines = file.readlines()

    output_lines = []

    for line in lines:
        output_lines += []

#is the same as generate_output_file method .. stopping
