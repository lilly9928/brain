import csv
import pandas as pd
def make_txt_file(output_dir,image_info_dir,caption_dir):

    image_info = open(image_info_dir,'r')
    read_image_info = pd.read_csv(image_info)
    read_image_info=read_image_info.sort_values(by=['case name'])
    read_image_info = read_image_info.reset_index(drop=True)
    caption_info = pd.read_excel(caption_dir, usecols=['image','caption'])
    caption_data_length = len(caption_info)
    caption_index = 0

    f = open(output_dir + '/captions.txt', 'w')
    f.writelines('image,caption,label\n')

    for i in range(len(read_image_info)):
        if caption_index == caption_data_length:
            break

        case_name = read_image_info['case name'][i]
        caption_case_name = caption_info['ID'][caption_index]

        if case_name == caption_case_name:
            image_name = str(read_image_info['cropped img'][i].split('.')[0])+'.jpg'
            caption = str(caption_info['str'][caption_index]).replace('\n',' ')
            label = str(read_image_info['class'][i])
            print(image_name,caption)
            f.write(image_name+','+caption+','+label+'\n')
        else:
            caption_index+=1
            print(caption_index)

    f.close()
    # print(caption_info['str'][0].replace('\n',' '))

if __name__ == '__main__':
    output_dir ='./'
    image_info_dir = 'D:/data/brain/preprocessed_data_v2/image_info.csv'
    caption_dir ='./iu_xray_data.xlsx'
    make_txt_file(output_dir,image_info_dir,caption_dir)