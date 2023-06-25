# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 16:48:19 2022

@author: Ozcan lab
"""


import cv2
import numpy as np
import math
import os
from os import listdir, mkdir
from os.path import isfile, join, isdir

MARGIN = 50
MASK = np.zeros((1, 1))
THRESHOLD = 10

def draw_alignmentspots(im, im_name, imagePath, alignment_circles):

    for (x, y, r) in alignment_circles:
        cv2.circle(im, (x, y), r, (0, 255, 0), 4)
        
    cv2.imwrite(imagePath + 'alignment_circles/' + im_name + '_circled.jpg', im)
        
def localize_spots(im, im_name, imagePath, alignmentSpotMap, MIN_X, MIN_Y):
    
    if not isdir(imagePath + 'alignment_circles/'):
            mkdir(imagePath + 'alignment_circles/')
    ind = 0
    all_circles = np.array([])       
    #for i in alignmentSpotMap.keys():
       # coords = alignmentSpotMap[i]
    #cropped = (im/256).astype('uint8')
    cropped =im #y,x
    #output = cropped
    
    #cropped = cropped[:, :, 1]
    cropped = cv2.medianBlur((cropped/256).astype('uint8'), 15)
    
    os.chdir(imagePath + 'alignment_circles/')
    
    cv2.imwrite(imagePath + 'alignment_circles/' + im_name + '_cropped' + '.tiff', cropped)
    
    circles = cv2.HoughCircles(cropped, cv2.HOUGH_GRADIENT, 1, 50, param1 = 5, param2 = 30, minRadius = 30, maxRadius = 50)
    circles = np.squeeze(circles)

    #draw_alignmentspots(cropped, im_name, imagePath, circles)
    
       
    if circles is not None:
        
        
        #circles = np.squeeze(circles)
        circles = np.round(circles).astype("int")
        #print('num:' + str(len(circles)))
        circle_n = []
        circle_c = []
        circle_dict = {}
        for i in alignmentSpotMap.keys():
            minDist = 100*100 #distance squared radius 100px
            minItem = []
            for circle in circles:
                coords = alignmentSpotMap[i]
                dist = np.abs(coords[0] - (circle[0]+MIN_X))*np.abs(coords[0] - (circle[0]+MIN_X)) + np.abs(coords[1] - (circle[1]+MIN_Y))*np.abs(coords[1] - (circle[1]+MIN_Y))
                
                if(dist < minDist):
                    minItem_cropped = circle
                    temp = circle.copy()
                    temp[0] += MIN_X
                    temp[1] += MIN_Y
                    minItem = temp
                    minDist = dist
            print(dist)
            if(minDist < 100*100):
                #print(minItem)
                # print(minDist)
                circle_n.append(minItem) 
                circle_c.append(minItem_cropped)
                circle_dict[i] = minItem
            else:
                print('template missed')
                print('ATTETION')
                print('ATTETION')
                return 0
        #   if circle[2]<45 and circle[2]>30:
                #print('x:'+str(circle[0])+' y:'+str(circle[1]))
                #dist = (circle[0]-500)*(circle[0]-500)+(circle[1]-500)*(circle[1]-500)
                #print(dist)
                #print(circle[2])
               # if dist<120000:
        #print('num_n:' + str(len(circle_n)))
        circles = np.array(circle_n)
        circles_c = np.array(circle_c)
        #circles_array = np.concatenate(circles), axis = 0)
        circles_array = circles
        #print(circles)
        #print(circles_array)
        if len(all_circles) == 0:
            all_circles = circles_array
        else:
            all_circles = np.concatenate((all_circles, circles_array), axis = 0)
        
        im_nmae_1 = im_name + '_cropped'
        
        draw_alignmentspots(cropped, im_nmae_1, imagePath, circles_c)
        
    else:
        print(circles) 
        
    ind += 1
        
    
    # #form final array with the aligned spots
    # circles_final = np.zeros((len(alignmentSpotMap.keys()), 3))
    # ind = 0
    # for i in alignmentSpotMap.keys():
    #     coords = alignmentSpotMap[i]
    #     circles_toprocess = []
    #     print(coords)
    #     for spot_ind in range(len(all_circles)):
    #         cur_circle = all_circles[spot_ind, :]
    #         if (coords[1] - cur_circle[1])**2 + (coords[0] - cur_circle[0])**2 <= (2*MARGIN)**2:
    #             circles_toprocess.append(cur_circle)
    #     if len(circles_toprocess) > 0:
    #         circles_final[ind,:] = np.mean(circles_toprocess, axis = 0)
    #     else:
    #         print('No spots detected for alignment spot ' + str(i))
    #     ind += 1
        
    # circles_final = np.round(circles_final).astype("int")
    # print(circles_final)
    circles_final = all_circles
    
    
    
    
    
    if len(circles_final) != len(alignmentSpotMap.keys()):
        print('Please select different alignment spots')
        
    return circle_dict, circles_final
        
                
def localizeWithCentroid(rotatedandscaled, coord_dict, display_spots):
    '''This function detects the actual center of a given spot by calculating the centroid of the pixel intensities. Then it updates the pointmap. This is much more effective than the Hough transform approach, but a bit slower.'''
    checkLocalization = False
    output_dict = {}
    for i in coord_dict.keys():
        try:
            coords = coord_dict[i]
            #print(coords)
            #print(type(rotatedandscaled[0,0]))
            cropped = rotatedandscaled[coords[1] - MARGIN:coords[1] + MARGIN, coords[0] - MARGIN: coords[0] + MARGIN]
            
            #if (i =='1') or (i=='16'):
            #   cropped = 256*256-1-cropped
            
            #if (i !='4') or (i!='5') or (i!='12') or (i!='13'):
            #    cropped = 256*256-1-cropped    
            #cropped = 256*256-1-cropped
            # image = cv2.medianBlur(image,5)
            #cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            avg = np.mean(np.ravel(cropped))
            ret, thresh = cv2.threshold(cropped, avg, 256*256-1, 0)
            M = cv2.moments(thresh)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if (display_spots == True):
                maxIntensity = 255.0  # depends on dtype of image data
                x = np.arange(maxIntensity)
                phi = .8
                theta = 5
                newImage0 = (maxIntensity / phi) * (cropped / (maxIntensity / theta)) ** 0.5
                newImage0 = np.array(newImage0, dtype=np.uint8)
                cv2.circle(newImage0, (cX, cY), 70, (0, 255, 0), 2)
                cv2.circle(newImage0, (cX, cY), 2, (0, 0, 255), 3)
                cv2.imshow('detected circles', newImage0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite('spot{}.jpg'.format(i), newImage0)

            output_dict[i] = (coords[0] + (cX - MARGIN), coords[1] + (cY - MARGIN))
            

        except:
            checkLocalization = True
            continue
    if (checkLocalization):
        print('\t\t^^Check localization^^')
    return output_dict



def rotateandscale_fromSpots(img, imagePath, im_name, degreesCCW, scaleValue, MIN_X, MIN_Y, alignment_spots_dict, correct_spot4_coordinates, key):
    '''
    :param img: the image that will get rotated and returned
    :param scaleFactor: option to scale image
    :param degreesCCW: DEGREES NOT RADIANS to rotate CCW. Neg value will turn CW
    :return: rotated image
    '''
    spot4 = alignment_spots_dict[key]
    
   # print(spot4)
    #print(correct_spot4_coordinates)
    (oldY, oldX) = (img.shape[0], img.shape[1])  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                                scale = scaleValue)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleValue, oldY * scaleValue
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # the warpAffine function call, below, basically works like this:
    # 1. apply the M transformation on each pixel of the original image
    # 2. save everything that falls within the upper-left "dsize" portion of the resulting image.

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty
    
    spot4_center = np.zeros((oldY, oldX))
    spot4_center[spot4[1], spot4[0]] = 255
    
    rotatedImg = cv2.warpAffine(img, M, dsize=(int(newX), int(newY)))
    
    rotated_spot4 = cv2.warpAffine(spot4_center, M, dsize=(int(newX), int(newY)))
    rotated_spot4 = np.uint8(rotated_spot4)
    newspot4_coords = np.nonzero(rotated_spot4)
    
    rotated_spot4_x = np.int(np.round(np.mean(newspot4_coords[1][:])))
    rotated_spot4_y = np.int(np.round(np.mean(newspot4_coords[0][:])))
                              
    cv2.circle(rotated_spot4, (rotated_spot4_x, rotated_spot4_y), spot4[2], 255, 4)
    cv2.rectangle(rotated_spot4, (rotated_spot4_x - 5, rotated_spot4_y - 5), (rotated_spot4_x + 5, rotated_spot4_y + 5), 255, -1)
   
    # Shifting the image

    shiftBy_x = correct_spot4_coordinates[0] - rotated_spot4_x
    shiftBy_y = correct_spot4_coordinates[1] - rotated_spot4_y
    
    num_rows, num_cols = rotatedImg.shape[:2]
    translation_matrix = np.float32([[1, 0, shiftBy_x], [0, 1, shiftBy_y]])
    final_image = cv2.warpAffine(rotatedImg, translation_matrix, (int(newX), int(newY)))
    
    
    #cv2.imwrite(imagePath + 'alignment_circles/' + im_name + '_rotated.jpg', rotatedImg)
    cv2.imwrite(imagePath + 'alignment_circles/' + im_name + '_aligned.jpg', (final_image/256).astype('uint8'))

    return final_image

def findAngle_and_ScaleFactor_spots(alignment_spots_dict, alignmentSpotMap, final_height, final_width):
    '''
    This function finds the corresponding angle between the pairs of alignment spots and turns the image to make it equal to 45 degree
    '''
    
    angleArr = []
    scaleArr = []
    cnt = 0
    prev_template = []
    prev_found = []
    for i in alignmentSpotMap.keys():
        if cnt == 0:
            coords_template = alignmentSpotMap[i]
            coords_found = alignment_spots_dict[i]
            prev_template = coords_template
            prev_found = coords_found
        else:
            coords_template = alignmentSpotMap[i]
            coords_found = alignment_spots_dict[i]
            grad_t = (coords_template[0]-prev_template[0])/(coords_template[1]-prev_template[1])
            dist_t = np.sqrt((coords_template[0]-prev_template[0])*(coords_template[0]-prev_template[0])+(coords_template[1]-prev_template[1])*(coords_template[1]-prev_template[1]))
            grad_f = (coords_found[0]-prev_found[0])/(coords_found[1]-prev_found[1])
            dist_f = np.sqrt((coords_found[0]-prev_found[0])*(coords_found[0]-prev_found[0])+(coords_found[1]-prev_found[1])*(coords_found[1]-prev_found[1]))
            scale = dist_t/dist_f
            angle = math.atan((grad_f-grad_t)/(1+grad_f*grad_t))
            angleArr.append(angle)
            scaleArr.append(scale)
            prev_template = coords_template
            prev_found = coords_found
        cnt += 1
   
    angle_avg = sum(angleArr)/len(angleArr)
    scale_avg = sum(scaleArr)/len(scaleArr)
    angle = angle_avg
    av_scale = scale_avg
        
        
    # spot1 = alignment_spots[0,0:2]
    
    # spot2 = alignment_spots[1,0:2]
    
    # spot3 = alignment_spots[2,0:2]
    
    # spot4 = alignment_spots[3,0:2]
    
    
    
    # x12 = np.abs(spot1[0] - spot2[0])
    # y12 = np.abs(spot2[1] - spot1[1])
    
    # x23 = np.abs(spot2[0] - spot3[0])
    # y23 = np.abs(spot3[1] - spot2[1])
    
    # x34 = np.abs(spot3[0] - spot4[0])
    # y34 = np.abs(spot4[1] - spot3[1])
    
    # x14 = np.abs(spot1[0] - spot4[0])
    # y14 = np.abs(spot4[1] - spot1[1])
    
    # angle12 = math.atan(x12/y12)
    # angle34 = math.atan(x34/y34)
    # angle = angle14 = math.atan(x14/y14)-np.pi/2
    # angle = (angle12 + angle34 + angle14)/3
    
    # x_scale = final_width/np.mean([x12, x23, x34])
    # y_scale = final_height/np.mean([y12, y23, y34])
    
    # av_scale = (x_scale + y_scale)/2 
    
    angleToRotate = -1 * angle * (180 / math.pi)
    return angleToRotate, av_scale


def generateMask(r):
    global MASK
    x, y = r, r
    MASK = np.ones((2*r,2*r))
    MASK[y][x] = 0
    for xs in range(r):
        for ys in range(r):
            if (0 < xs ** 2 + ys ** 2 < r ** 2):
                MASK[y + ys][x + xs] = 0
                MASK[y - ys][x - xs] = 0
                MASK[y + ys][x - xs] = 0
                MASK[y - ys][x + xs] = 0
    return MASK

def getStats(image, r, center, commands, visualize, threshold, directory, image_name):
    x, y = center[0], center[1]
    thisSpot = np.array(image)[y-r:y+r, x-r:x+r]
    thisSpot = np.ma.array(thisSpot, mask=MASK)
    ravelled = thisSpot.compressed()
    refStd, refMean, refMax = np.std(ravelled), np.mean(ravelled), max(ravelled)
    thisSpot = np.ma.masked_outside(thisSpot, refMean - threshold[0] * refStd, refMean + threshold[0] * refStd)
    refMax = max(thisSpot.compressed())
    #thisSpot = np.ma.masked_greater(thisSpot, threshold[1] * refMean)
    #thisSpot = np.ma.masked_less(thisSpot, threshold[2] * refMean)
 
    if (visualize):
        to_show = np.concatenate(((thisSpot.filled(256*256-1)/256).astype('uint8'), (image[y-r:y+r, x-r:x+r]/256).astype('uint8')), axis=1)
        #cv2.imshow('spot mask vs original', to_show)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #os.chdir(directory)
        cv2.imwrite(directory + image_name+str(x)+'_'+str(y)+'.jpg', to_show)
    thisSpot = thisSpot.compressed()
    options = [np.std, np.mean, np.amax, np.amin]
    result = [options[c](thisSpot) for c in range(len(commands)) if commands[c] != 0]
    return result


def drawCirclesAndLabels(image, pointMap, radius_to_draw):
    '''
    This function is just to display the image with the labels that we predetermined,
    it has no impact on the resulting calculations
    :param image: The image that is going to be drawn on. This is the image that is ALREADY ALIGNED.
    :param pointMap: The pointmap that we have predefined
    :param radius_to_draw: The radius of each circle that will be drawn
    :return:
    '''

    for key, value in pointMap.items():

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2

        if key not in ['A', 'B', 'C', 'D']:
            cv2.circle(image, value, radius_to_draw, color, thickness)

        cv2.putText(image, key, value, font,
                                fontScale, color, thickness, cv2.LINE_AA)
    return image
