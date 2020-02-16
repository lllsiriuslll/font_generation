#!/bin/sh

CURRENT=`dirname $0`
PARENT=`dirname ${CURRENT}`
cd ${PARENT}
FONTS_DIR_NAME="font"
SAVE_IMAGES_DIR_NAME="font_images"
SAVE_OBJ_DIR_NAME="font_obj"
fonts_path="${PARENT}/${FONTS_DIR_NAME}"
save_img_path="${CURRENT}/${SAVE_IMAGES_DIR_NAME}"
save_obj_path="${CURRENT}/${SAVE_OBJ_DIR_NAME}"

fonts=`find $fonts_path -maxdepth 1 -type f`

COUNT=0
src_font="${fonts_path}/src_font.ttc"
echo "Font to Image Start"
for dst_font in $fonts;
do
    if [ $dst_font != $src_font ]; then
        python font2img.py --src_font $src_font \
                           --dst_font $dst_font \
                           --charset JP_KANA \
                           --sample_count 1000 \
                           --sample_dir $save_img_path \
                           --label $COUNT \
                           --filter 1 \
                           --shuffle 1 #\
                           #--canvas_size 512 \
                           #--char_size 400
    fi
    COUNT=`expr $COUNT + 1`
done
echo "Font to Image Done"

echo "Image to Package Start"
python package.py --dir $save_img_path \
                  --save_dir $save_obj_path \
                  --split_ratio 0.1
echo "Image to Package Done"

echo "Hit Enter Key."
read Wait
