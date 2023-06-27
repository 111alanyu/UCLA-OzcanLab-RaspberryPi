import cv2
import numpy as np
import csv
import time
from helpers_timelapse_decap import drawCirclesAndLabels, \
   localizeWithCentroid, getStats, generateMask, localize_spots, findAngle_and_ScaleFactor_spots, rotateandscale_fromSpots, draw_alignmentspots

# This is for reading the images that are in the fluorescent/ directory
from os import listdir, mkdir
from os.path import isfile, join, isdir

command_dict = {0: 'std', 1: 'mean', 2: 'max', 3: 'min'}
alignmentSpotMap = {
    '4': (820-50, 250+100),
    '5': (360-50, 425+100),
    '12': (830-50, 570+100),
    '13': (370-50, 745+100),
}

KEY = '4' #always set to the spot that we are doing localziation on.
correct_spot4_coordinates = [820-50, 250+100]

col_width = 150 #not used
row_height = col_width


# #### THIS PART IS ESSENTIAL, THIS IS OUR SPOT MAP THAT OUR ENTIRE PROGRAM IS BASED ON
# pointMapInit = {
#     '1': (1470, 1780), 
#     '2': (1805, 1795),
#     '3': (2135, 1795),
#     '4': (2460, 1795),
#     '5': (1475, 2120), 
#     '6': (1800, 2120),
#     '7': (2130, 2125),
#     '8': (2460, 2125),
#     '9': (1470, 2455),
#     '10': (1800, 2445),
#     '11': (2130, 2450),
#     '12': (2460, 2455),
#     '13': (1465, 2785),
#     '14': (1800, 2775),
#     '15': (2125, 2780), 
#     '16': (2455, 2785),
#     '17': (1960, 2280)
# }
#### THIS PART IS ESSENTIAL, THIS IS OUR SPOT MAP THAT OUR ENTIRE PROGRAM IS BASED ON
pointMapInit = {
    '1': (350-50, 255+100), 
    '2': (505-50, 260+100),
    '3': (660-50, 250+100),
    '4': (820-50, 250+100),
    '5': (360-50, 425+100),
    '6': (515-50, 420+100),
    '7': (670-50, 415+100),
    '8': (825-50, 410+100),
    '9': (360-50, 585+100),
    '10': (520-50, 575+100),
    '11': (675-50, 575+100),
    '12': (830-50, 570+100),
    '13': (370-50, 745+100),
    '14': (520-50, 735+100),
    '15': (680-50, 730+100), 
    '16': (835-50, 725+100),
    '17': (590-50, 500+100)
}

NUM_SPOTS = len(pointMapInit) - 4
NUM_STATS = len(command_dict)

# Note: if you change this, only change the file name of
# the template, dont change the keys of this dictionaryD:\Artem\VFA\CPN\70_nm_CPN\sample_incubation\0.5
# e.g. Do not change 'template_A'
template_dictionary = {
    'template_A': 'alignment_A_green.jpg',
    'template_B': 'alignment_B_green.jpg',
    'template_C': 'alignment_C_green.jpg',
    'template_D': 'alignment_D_green.jpg'
}

# CONSTANTS
MASK_RADIUS = 30



# CROP IMAGE DIMENSIONS CONSTANTS
YMIN_BOUND = 600
YMAX_BOUND = 1700
XMIN_BOUND = 1100
XMAX_BOUND = 2300

THRESH1 = 3 #Acceptable margin for the first filter (in multiples of the standard deviation)
THRESH2_UPPER = 1.5 #All pixels lower than THRESH2_UPPER*max_in_spot are kept
THRESH2_LOWER = 0 #All pixels greater than THRESH2_LOWER*max_in_spot are kept
THRESH = [THRESH1, THRESH2_UPPER, THRESH2_LOWER]
show_mask = 1

def getCircleData(imagePath, image_name, image_num, current_Point_Map, displayCirclesBool, whichCommands, r, angleToRotate, scaleValue, alignment_spots_dict):
    '''
    :param imagePath: the image path for the image that we are going to find all the averages for
    :param image_name: This is the name of the image that we are going to find all the averages for
    :param displayCirclesBool: Determines if the images will be displayed on the screen, or saved to a file

        True: The images will be output to the screen

        False: The images will be saved to a directory named "processed" inside
        the directory the user specified at the beginning
    :param whichCommand: Indicates which statistic is being requested

    :return:
    '''

    # This output array will be returned and will be a row in the csv file
    output = []
    output.append([image_name for i in range(NUM_STATS - whichCommands.count(0))])
    # Import (and convert from DNG if necessary)
    full_image_path = imagePath + image_name

    

    image = cv2.imread(full_image_path,-1)
    #include only for RGB input images
    if image_num == 0:
        
        image = image[:, :, 1]
    else:
        image = image[:, :, 1]
        
    #image = image[:, :, 1]
    #include if you need rotation
    image = cv2.rotate(image, cv2.ROTATE_180)
    
    image_o = image.copy()
    image = image[YMIN_BOUND: YMAX_BOUND, XMIN_BOUND: XMAX_BOUND]
    if image_num == 0:
    # Crop and align
        alignment_spots_dict = {}
        
        alignment_spots_dict, alignment_spots = localize_spots(image, image_name, imagePath, alignmentSpotMap, 0, 0)
        
        #image_spots = image_o
        
        #draw_alignmentspots(image_spots, image_name, imagePath, alignment_spots)
        
        angleToRotate, scaleValue = findAngle_and_ScaleFactor_spots(alignment_spots_dict,alignmentSpotMap, col_width, row_height)
    
        aligned_image = rotateandscale_fromSpots(image, imagePath, image_name, angleToRotate, scaleValue, XMIN_BOUND, YMIN_BOUND, alignment_spots_dict, correct_spot4_coordinates, KEY)

    # Improve localization
    
        pointMap = localizeWithCentroid((aligned_image/256).astype('uint8'), current_Point_Map, False)
    else:
        aligned_image = rotateandscale_fromSpots(image, imagePath, image_name, angleToRotate, scaleValue, XMIN_BOUND, YMIN_BOUND, alignment_spots_dict, correct_spot4_coordinates, KEY)
        
        pointMap = current_Point_Map
        
        


    ## Grayscale the image to begin masking process
    #  aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
    #red_channel = aligned_image[:, :, 1]
    red_channel = aligned_image
    image_name = image_name.split('.')[0]
    for key, value in pointMap.items():

        # We do not need to print the average intensity for the alignment markers
        
        # mask = create_circular_mask(h, w, MASK_RADIUS, value[0], value[1])
        # maskedImage = np.multiply(aligned_image, mask)

        # averageIntensity = findAverageLightIntensity(maskedImage, mask)
        if not isdir(imagePath + 'spot_masks/'):
            mkdir(imagePath + 'spot_masks/')
        output.append(getStats(red_channel, r, value, whichCommands, show_mask, THRESH, imagePath + 'spot_masks/', image_name))
        
    # Display or save
    
    if displayCirclesBool == True:
        labeled_image = drawCirclesAndLabels((aligned_image/256).astype('uint8'), pointMap, r)
        cv2.imshow("Labeled Circles for " + image_name, labeled_image)

    else:
        labeled_image = drawCirclesAndLabels((aligned_image/256).astype('uint8'), pointMap, r)
        if not isdir(imagePath + 'processed_jpgs/'):
            mkdir(imagePath + 'processed_jpgs/')
        cv2.imwrite(imagePath + 'processed_jpgs/' + image_name + '_r='+str(r)+'_processed.jpg', labeled_image)

    # Prints the path for the image that was just processed
    print("\tFinished Processing: " + full_image_path)
    return output, pointMap, angleToRotate, scaleValue, alignment_spots_dict


def averagesOfAllImages(displayCirclesBool=False, test_directory_name="", stat_commands='1100', r=MASK_RADIUS):
    '''

    This function simply runs the findAllCircleAveragesFor every image in our list.

    The list is compiled by first prompting the user for the name of the directory they would like to run a test on.
    It then finds all the tif images inside of that directory and adds them to the list.

    NOTE: THE REPOSITORY THE USER ENTERS MUST BE INSIDE 'datasets' DIRECTORY.

    For each image, once it receives the intensity of each spot, it takes the information
    and writes it to a csv file inside of the user-specified directory.

    E.g. Say that we have a directory named 'tiff-conv1' inside the 'datasets' directory
        Once the processing is done, inside 'datasets/tiff-conv1' there will be a csv file named 'tiff-conv1.csv'
        containing all the informatino that we found


    :param displayCirclesBool:
    :return:
    '''
    imageList = []

    if test_directory_name[-1] != '/':
        test_directory_name += '/'
        test_directory_path = test_directory_name

    ##Asserting that the directory input by user is valid and has images ending with .tif inside of it
    if (isdir(test_directory_path)):
        imageList = [f for f in listdir(test_directory_path) if (isfile(join(test_directory_path, f))) and (
                    f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.dng') or f.endswith('.tiff'))]
        if (len(imageList) == 0):
            print("\tError: No images here, please check the directory")
            return
    else:
        print("\tError: Invalid directory")
        return

    imageList = sorted(imageList)
    print('\t' + str(len(imageList)) + ' images imported...')

    if stat_commands.isnumeric() and len(stat_commands) == NUM_STATS:
        stat_commands = [int(c) for c in stat_commands]
    else:
        print('\tInvalid metric settings, type \'help\'...')
        return

    start = time.time()
    ##### Writes data acquired from list to our csv file
    i = 0
    matrix = np.ones((NUM_SPOTS+2, NUM_STATS - stat_commands.count(0)))
    generateMask(r)
    
    #im3 = imageList[12]
    #del imageList[12]
    #imageList = [im3] + imageList
    #print(imageList)
    for image in imageList:
        if i == 0:
            current_Point_Map = pointMapInit
            image_num = i
            signals, PointMap, angleToRotate, scaleValue, alignment_spots_dict = getCircleData(test_directory_path, image, image_num, current_Point_Map, displayCirclesBool, stat_commands, r, None, None, None)
            matrix = [signals]
            i += 1
            continue
        else:
            image_num = i
            current_Point_Map = PointMap
            signals, PointMap, angleToRotate, scaleValue, alignment_spots_dict = getCircleData(test_directory_path, image, image_num, current_Point_Map, displayCirclesBool, stat_commands, r, angleToRotate, scaleValue, alignment_spots_dict)
            matrix += [signals]
            i += 1
            
    if not isdir(test_directory_path + 'csv/'):
        mkdir(test_directory_path + 'csv/')
    j = 0

    matrix = np.asarray(matrix)
    print(matrix)
    
    for s in range(NUM_STATS):
        if stat_commands[s] != 0:
            with open(test_directory_path + 'csv/' + command_dict[s] + '_r=' + str(r) + '_fixed.csv', 'w+', newline='') as f:
                #thisMatrix = np.vstack([np.append(np.arange(NUM_SPOTS + 1), 'Error Flag'), matrix[:, :, j]])
                writer = csv.writer(f)
                writer.writerows(np.squeeze(matrix[:, :, j]))
            j += 1
    end = time.time()
    print('Average runtime: ' + str((end - start) / len(imageList)))


def main():
    global show_mask
    while (1):
        folder_name = input('Enter directory to test, \'quit\' to exit, \'help\' for more info: ')
        if folder_name == 'quit':
            return
        elif folder_name == 'help':
            print('\tThe image folder should exist within the same directory the script is run from.\n' +
                '\tTo toggle statistics, enter a string of binary digits ' +
                '(separated by a comma) corresponding to\n' +
                '\t\t[Std, Mean, Max, Min]\n' +
                '\tFor example, entering \'this_folder, 1100\' ' +
                'returns the Standard Deviation and the Mean.\n\tMean is taken by default (\'0100\').\n' +
                '\tTo set the radius, type \'r=[your value here]\' separated by a comma. Ex: \'folder,r=45,1111\'.\n\tThe default radius is 60px.\n' +
                '\tTo display spot masks as they\'re generated, type \'mask=1\'. Turn it off with \'mask=0\'.')
        # Change to true to display images with circles drawn on
        else:
            folder_name = (folder_name.replace(" ", "")).split(',')
            if len(folder_name) >= 2:
                extra_commands = folder_name[1:]
                radius = MASK_RADIUS
                metrics = '0100'
                for com in extra_commands:
                    if com[:2] == 'r=':
                        if com[2:].isnumeric():
                            radius = int(com[2:])
                        else:
                            print('\tInvalid radius. Type \'help\'')
                    elif com[:5] == 'mask=':
                        if com[5].isnumeric() and len(com)==6:
                            show_mask = int(com[5])
                        else:
                            print('\tInvalid mask setting. Type \'help\'')
                    elif com.isnumeric():
                        metrics = com
                    else:
                        print('\tInvalid setting input. Type \'help\'')
                averagesOfAllImages(False, folder_name[0], metrics, int(radius))
            else:
                averagesOfAllImages(False, folder_name[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main_ui_analysis(args):
    global show_mask
    while (1):
        folder_name = args
        if folder_name == 'quit':
            return
        elif folder_name == 'help':
            print('\tThe image folder should exist within the same directory the script is run from.\n' +
                '\tTo toggle statistics, enter a string of binary digits ' +
                '(separated by a comma) corresponding to\n' +
                '\t\t[Std, Mean, Max, Min]\n' +
                '\tFor example, entering \'this_folder, 1100\' ' +
                'returns the Standard Deviation and the Mean.\n\tMean is taken by default (\'0100\').\n' +
                '\tTo set the radius, type \'r=[your value here]\' separated by a comma. Ex: \'folder,r=45,1111\'.\n\tThe default radius is 60px.\n' +
                '\tTo display spot masks as they\'re generated, type \'mask=1\'. Turn it off with \'mask=0\'.')
        # Change to true to display images with circles drawn on
        else:
            folder_name = (folder_name.replace(" ", "")).split(',')
            if len(folder_name) >= 2:
                extra_commands = folder_name[1:]
                radius = MASK_RADIUS
                metrics = '0100'
                for com in extra_commands:
                    if com[:2] == 'r=':
                        if com[2:].isnumeric():
                            radius = int(com[2:])
                        else:
                            print('\tInvalid radius. Type \'help\'')
                    elif com[:5] == 'mask=':
                        if com[5].isnumeric() and len(com)==6:
                            show_mask = int(com[5])
                        else:
                            print('\tInvalid mask setting. Type \'help\'')
                    elif com.isnumeric():
                        metrics = com
                    else:
                        print('\tInvalid setting input. Type \'help\'')
                averagesOfAllImages(False, folder_name[0], metrics, int(radius))
            else:
                averagesOfAllImages(False, folder_name[0])
        return


if __name__ == '__main__':
    main()

