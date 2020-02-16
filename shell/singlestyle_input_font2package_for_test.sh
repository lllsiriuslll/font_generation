#!/bin/sh

CURRENT=`dirname $0`
PARENT=`dirname ${CURRENT}`
cd ${PARENT}
FONTS_DIR_NAME="font"
SAVE_IMAGES_DIR_NAME="test_font_images"
SAVE_OBJ_DIR_NAME="test_font_obj"
fonts_path="${PARENT}/${FONTS_DIR_NAME}"
save_img_path="${CURRENT}/${SAVE_IMAGES_DIR_NAME}"
save_obj_path="${CURRENT}/${SAVE_OBJ_DIR_NAME}"

fonts=`find $fonts_path -maxdepth 1 -type f`

COUNT=0
src_font="${fonts_path}/src_font.ttc"
echo "Font to Image Start"
for dst_font in $fonts;
do
    python font2img.py --src_font $dst_font \
                        --dst_font $dst_font \
                        --charset JP_KANA \
                        --sample_count 1000 \
                        --sample_dir "${save_img_path}_${COUNT}" \
                        --label $COUNT \
                        --filter 0 \
                        --shuffle 1 #\
                        #--canvas_size 512 \
                        #--char_size 400
    echo "Font to Image Done"
    echo "Image to Package Start"
    python package.py --dir "${save_img_path}_${COUNT}" \
                      --save_dir "${save_obj_path}_${COUNT}" \
                      --split_ratio 0.0
    echo "Image to Package Done"

    COUNT=`expr $COUNT + 1`
done
echo "Font to Image to Package Done"
echo "Hit Enter Key."
read Wait
