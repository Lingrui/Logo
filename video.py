#!/usr/bin/env python3 
import cv2
import numpy as np 
import os
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from skimage import measure
import sys
import pickle 
from PIL import ImageFont, ImageDraw, Image 
#set up the font###
font = ImageFont.truetype("./Helvetica-Regular.ttf",18)

dic = { 0:'Background',
        1:'PlayStation (Sony)',
        2:'Nissan',
        3:'Heineken',
        4:'Lays',
        5:'UniCredit',
        6:'Gazprom'
        }

## html text #
html_head = '<html><head><meta http-equiv="Content-Type" content="text/html; charset=windows-1252"></head><body style="font-family:helvetica">' + "\n"
html_head += '<h1>Video analysis of brand exposure</h1><br><video width="896" height="252" controls=""><source src="video.mp4" type="video/mp4">Your browser does not support the video tag.</video><br><hr>' + "\n"
html_head += '<h1>Individual brands detected</h1><table border="1">' + "\n"
html_head += '<tbody>'

html_tail = '</tbody></table>'

class Model:
    def __init__ (self, X, is_training, path, name):
        mg = meta_graph.read_meta_graph_file(path + '.meta')
        self.logits, = tf.import_graph_def(mg.graph_def, name=name,
                    input_map={'images:0': X, 'is_training:0': is_training},
                    return_elements=['logits:0'])
        if len(self.logits.get_shape()) == 4:
            # FCN
            self.is_fcn = True
            self.prob = tf.nn.softmax(self.logits)
            #self.prob = tf.squeeze(tf.slice(tf.nn.softmax(self.logits), [0,0,0,1], [-1,-1,-1,1]), 3)
        else:
            # classification
            self.is_fcn = False
            self.prob = tf.nn.softmax(self.logits)
        self.saver = tf.train.Saver(saver_def=mg.saver_def, name=name)
        self.loader = lambda sess: self.saver.restore(sess, path)
        pass
    pass

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input',None,'')
flags.DEFINE_string('output',None,'')
flags.DEFINE_integer('fps',30,'')
flags.DEFINE_string('model',None,'')
flags.DEFINE_integer('stride',16,'')
flags.DEFINE_string('name', 'logits:0', '')
flags.DEFINE_float('cth', 0.5, '')
flags.DEFINE_string('record', None, '')
flags.DEFINE_float('th', 0.5, '')
flags.DEFINE_float('prob', 0.001, 'cutoff of prob to show text annotation')
flags.DEFINE_integer('seglen',15,'')
flags.DEFINE_string('pickle', 'tmp.pkl','')
flags.DEFINE_string('html', 'brand_test.html','')

def save_prediction_image ( image, prob):
    # image: original input image
    # prob: probability
    #add weight #
    h,w, _ = prob.shape
    weight = 1 - np.abs(np.arange(h) - h/2)/h
    weight = np.reshape(weight,(h,1,1))

    prob1 = np.expand_dims(1 - prob[:, :, 0], axis=2)
    mask = (prob1 > FLAGS.th).astype(np.float32) # convert to 0/1 mask 
    prob = prob * mask
    p1 = np.sum(prob[:, :, 1:])  #sum pixel of not background
    prob = prob * weight
    score = []  # score = np.sum(prob, axes=[0,1])
    for i in range(prob.shape[2]):
        score.append(np.sum(prob[:,:,i])/(w*h))
        #print (np.sum(prob[:,:,i]/(w*h)))
    tmp = score[1:]
    if sum(tmp) == 0:
        cate = 0 
    else:
        cate = tmp.index(max(tmp)) + 1  ### max score category 

    H = max(image.shape[0], prob.shape[0])
    both = np.zeros((H, image.shape[1] + prob.shape[1], 3))
    both[0:image.shape[0],:image.shape[1], :] = image
    both[0:prob.shape[0],image.shape[1]:, :] = prob1 * image
    return both, score, cate, p1

def process_pkl():
    cate = 0 
    start = 0 
    end = 0 
    seg = []  ## save the frame number to be segment
    f_dic = {}
    html_i = []
    with open(FLAGS.record, 'rb') as f:
        ss = pickle.load(f)
        i = 0 
        s_list = []
        for item in ss:
            frame,ca,brand,s = item  #frame number, category, brand, score list of all brand 
            f_dic[frame] = [brand,s]
            if sum(s) > 0 :  #exist object
                #score = max(s)/sum(s)  # calculate the max porb
                score = max(s)
                cate_tmp = ca 
                if i > 0:
                    #if score > FLAGS.th:
                    if score > FLAGS.prob:
                        if cate_tmp == cate:
                            end = i 
                            s_list.append(score)
                        else:
                            if (end - start) > FLAGS.seglen:
                                score_sum = 0 
                                for num in range(start,end+1):
                                    seg.append(num)
                                    _,sss=f_dic[num]
                                    #score_sum += max(sss)/sum(sss)
                                    score_sum += max(sss)
                                html_i.append([start,end-start,cate,score_sum])

                            cate = cate_tmp
                            start = i
                            s_list = [score]
                    else:  # when score < FLAGS.th
                        if (end - start) > FLAGS.seglen:
                            score_sum = 0 
                            for num in range(start,end+1):
                                seg.append(num)
                                _,sss=f_dic[num]
                                #score_sum += max(sss)/sum(sss)
                                score_sum += max(sss)
                            html_i.append([start,end-start,cate,score_sum])
                        cate = ca
                        start = i 
                        end = i 
                        s_list = []
                else:   ### when i = 0 
                    #if score > FLAGS.th:
                    if score > FLAGS.prob:
                        cate = ca 
                        s_list = [score]
            else: #no ad brand detected
                if (end - start) > FLAGS.seglen:
                    score_sum = 0 
                    for num in range(start,end+1):
                        seg.append(num)
                        _,sss=f_dic[num]
                        #score_sum += max(sss)/sum(sss)
                        score_sum += max(sss)
                    html_i.append([start,end-start,cate,score_sum])
                cate = ca
                start = i
                end = i 
            i += 1
    return seg, f_dic,html_i

def html_brand (brand,start,dur,score,filename):
    text = "<tr><td><table><tbody><tr><td>Brand:</td><td>"+dic[brand]+"</td></tr>"+"\n"
    text += "<tr><td>Offset:</td><td>"+str(start)+"s</td></tr>"+"\n"
    text += "<tr><td>Duration:</td><td>"+str(dur)+"s</td></tr>"+"\n"
    text += "<tr><td>IDS Score:</td><td>"+str(score)+"</td></tr>"+"\n"
    text += "</tbody></table></td><td>" +"\n"
    text += '<video width="896" height="252" controls=""><source src="'+str(filename)+'.mp4" type="video/mp4">Your browser does not support the video tag.</video></td></tr>'+'\n'
    return text

def main(_):
    X = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="images")
    is_training = tf.placeholder(tf.bool, name="is_training")
    model = Model(X, is_training, FLAGS.model, 'xxx')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    video_input = cv2.VideoCapture(FLAGS.input)
    video_output = None
    scores = [] 
    ##read in the record
    if FLAGS.record is not None:
        seg, f_dic, html_i = process_pkl()  # get frame numbers
        with open(FLAGS.html,'w') as f:
            f.write(html_head)
            mp4_i = 0
            for slide in html_i:
                start,dur,brand,sc = slide 
                f.write(html_brand(brand,'%.2f' %(start/FLAGS.fps),'%.2f' %(dur/FLAGS.fps),'%.2f' %sc,mp4_i)) 
                mp4_i += 1
            f.write(html_tail)
    with tf.Session(config=config) as sess:
        model.loader(sess)
        frame = 0 ## frame number of the vedio 
        while video_input.grab():
            flag,image = video_input.retrieve()
            if not flag:
                break
            H, W, _ = image.shape 
            if model.is_fcn:
                H = H // FLAGS.stride * FLAGS.stride
                W = W // FLAGS.stride * FLAGS.stride
                image = image[:H, :W, :]
            batch = np.expand_dims(image, axis=0).astype(dtype=np.float32)
            prob = sess.run(model.prob, feed_dict={X: batch, is_training: False})
            #print (prob.shape)
            if video_output is None:
                video_output = cv2.VideoWriter(FLAGS.output,cv2.VideoWriter_fourcc(*"MJPG"),FLAGS.fps,(W*2,H))
            if len(prob.shape) == 1:
                print(prob[0])
            else:
                pred, score, cate, ppp = save_prediction_image(image,prob[0])
                scores.append([frame,cate,dic[cate],score])
                if sum(score[1:]) > 0:
                    if FLAGS.record is None:  ##print scores on all images
                        #convert the image to RGB
                        cv2_in_rgb = cv2.cvtColor(pred.astype(np.uint8),cv2.COLOR_BGR2RGB)
                        #pass the image to PIL
                        pil = Image.fromarray(cv2_in_rgb)
                        draw = ImageDraw.Draw(pil)
                        #draw the text
                        for i in range(1,7):
                            if i == cate:
                                text1 = str(dic[cate]) + ":"
                                #text2 = '%.2f' % (score[i]/sum(score[1:]))
                                text2 = '%.3f' % (score[i])
                                draw.text((660,180+i*20),text1,font = font,fill = (0,255,0) )
                                draw.text((820,180+i*20),text2,font = font,fill = (0,255,0))
                            else:
                                text1 = str(dic[i]) + ":"
                                #text2 = '%.2f' % (score[i]/sum(score[1:]))
                                text2 = '%.3f' % (score[i])
                                draw.text((660,180+i*20),text1,font = font,fill = (255,255,255))
                                draw.text((820,180+i*20),text2,font = font,fill = (255,255,255))
                        draw.text((660,320),"detect area: ",font = font,fill = (255,255,255))
                        draw.text((820,320),str(ppp),font = font,fill = (255,255,255))
                        #get back the image to OpenCV
                        cv2_in_processed = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) 
                        video_output.write(cv2_in_processed.astype(np.uint8))
                        pass
                    else:
                        if frame in seg and ppp > 500:
                            #convert the image to RGB
                            cv2_in_rgb = cv2.cvtColor(pred.astype(np.uint8),cv2.COLOR_BGR2RGB)
                            #pass the image to PIL
                            pil = Image.fromarray(cv2_in_rgb)
                            draw = ImageDraw.Draw(pil)
                            #draw the text
                            for i in range(1,7):
                                if i == cate:
                                    text1 = str(dic[cate]) + ":"
                                    #text2 = '%.2f' % (score[i]/sum(score[1:]))
                                    text2 = '%.3f' % (score[i])
                                    draw.text((660,180+i*20),text1,font = font,fill = (0,255,0) )
                                    draw.text((820,180+i*20),text2,font = font,fill = (0,255,0))
                                else:
                                    text1 = str(dic[i]) + ":"
                                    #text2 = '%.2f' % (score[i]/sum(score[1:]))
                                    text2 = '%.3f' % (score[i])
                                    draw.text((660,180+i*20),text1,font = font,fill = (255,255,255))
                                    draw.text((820,180+i*20),text2,font = font,fill = (255,255,255))
                            #get back the image to OpenCV
                            cv2_in_processed = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR) 
                            video_output.write(cv2_in_processed.astype(np.uint8))
                            pass
                        else:
                            video_output.write(pred.astype(np.uint8))
                else:
                    video_output.write(pred.astype(np.uint8))
            frame += 1
            pass
    video_output.release()
    #os.system('sleep 60')
    if FLAGS.record is None:
        pickle.dump(scores,open(FLAGS.pickle,'wb'))
    else:
        os.system('ffmpeg -i %s -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p ./video.mp4' % FLAGS.output)
        os.system('sleep 400')
        j = 0 
        for html in html_i:
            start, duration , _ , _ = html
            os.system('ffmpeg -ss %s -i video.mp4 -t %s %s.mp4' % (frame2second(start), frame2second(duration), str(j) ))
            os.system('sleep 10')
            j += 1
    pass

def frame2second(frame):
    min = (frame//FLAGS.fps)//60
    sec = ((frame-FLAGS.fps*60*min)//30)
    milsec = '%.2f' % ((frame-FLAGS.fps*60*min- 30*sec)/float(FLAGS.fps))
    second = str(min)+":"+str(sec)+"."+str(milsec).replace("0.","")
    return second

if __name__ == '__main__':
    tf.app.run()
