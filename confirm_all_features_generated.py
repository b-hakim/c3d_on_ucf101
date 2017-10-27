import Util as utl

def confirm_features_generated(list_features_file_path):
    with open('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/rest_train_01.ls', 'w') as fw:
        with open(list_features_file_path) as file:
            for line in file:
                s = line.split()
                if s[0].lower().__contains__('apply'):
                    continue
                if not utl.is_file_exists(s[0]+'.prob.txt'):
                    print s[0]+'.prob.txt 0'
                elif not utl.is_file_exists(s[0]+'.conv5a.txt'):
                    print s[0] + '.conv5a.txt 0'
                elif not utl.is_file_exists(s[0]+'.conv5b.txt'):
                    print s[0] + '.conv5b.txt 0'
                elif not utl.is_file_exists(s[0]+'.fc6.txt'):
                    print s[0] + 'fc6.txt 0'
                elif not utl.is_file_exists(s[0]+'.fc7.txt'):
                    print s[0] + '.fc7.txt 0'
                elif not utl.is_file_exists(s[0]+'.fc8.txt'):
                    print s[0] + '.fc8.txt 0'
                elif not utl.is_file_exists(s[0]+'.pool5.txt'):
                    print s[0] + '.pool5.txt 0'

               # if not utl.is_file_exists(s[0] + '.prob.txt') \
               # or not utl.is_file_exists(s[0] + '.conv5a.txt') \
               # or not utl.is_file_exists(s[0] + '.conv5b.txt') \
                #if not utl.is_file_exists(s[0] + '.fc6.txt'):
               # or not utl.is_file_exists(s[0] + '.fc7.txt') \
               # or not utl.is_file_exists(s[0] + '.fc8.txt') \
               # or not utl.is_file_exists(s[0] + '.pool5.txt'):
                    #fw.write(s[0] + " 0\n")

confirm_features_generated('/home/kasparov092/sources/c3d/v1.0/examples/c3d_feature_extraction/c3d_sports_finetuned_ucf_model/output_train_list_prefix.txt')