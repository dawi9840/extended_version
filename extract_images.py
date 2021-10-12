import os
import cv2
import glob


def camera_info(cap):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w: {w}, h: {h}')

    while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                cv2.imshow('Raw Video Feed', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    print('Done.')
    cap.release()
    cv2.destroyAllWindows()


def extract_images(cap, str_class):
    '''Need a floder ./resource/extract_images/[str_class] 
        to save extract images from a video.'''

    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    count = 0
    while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('./resource/extract_images/'+str_class+'/image_{0:0>3}.jpg'.format(count), frame) # Save frame as JPEG file.
                print(f'save frame: {count}')
                count += 1
                cv2.imshow('extract_video', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    print('Extract images done!')
    cap.release()
    cv2.destroyAllWindows()


def img_to_video(input_imgs_floder, output_video):
    ''''path = './*.jpg', output_video = file name.'''
    img_array = []

    for filename in glob.glob(input_imgs_floder):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print('create done!')


if __name__ == '__main__':

    # Directory
    directory = 'jumping_jacks'
    
    # Parent Directory path
    parent_dir = './resource/extract_images/'
    path = os.path.join(parent_dir, directory)
    
    # Create the directory in path.
    os.mkdir(path)

    extract_images(cap=cv2.VideoCapture('./jumping_jacks_23s.mp4'), str_class=directory)  