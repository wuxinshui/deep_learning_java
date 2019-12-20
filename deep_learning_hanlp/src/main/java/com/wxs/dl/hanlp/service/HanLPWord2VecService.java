/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-11-02 12:09</create-date>
 *
 * <copyright file="Demo.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.wxs.dl.hanlp.service;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.mining.word2vec.Word2VecTrainer;
import com.hankcs.hanlp.mining.word2vec.WordVectorModel;
import org.springframework.util.ResourceUtils;

import java.io.IOException;

/**
 * 演示词向量的训练与应用
 *
 * @author hankcs
 */
public class HanLPWord2VecService {
    //private static final String TRAIN_FILE_NAME = "D:\\git\\github\\ai\\deep_learning_java\\deep_learning_hanlp\\src\\main\\resources\\data\\sogou_segment_zh.txt";
    //private static final String MODEL_FILE_NAME = "D:\\git\\github\\ai\\deep_learning_java\\deep_learning_hanlp\\src\\main\\resources\\data\\sogou_segment_zh_model";
    //相对地址
    private static final String TRAIN_FILE_NAME = "sogou_segment_zh.txt";
    private static final String MODEL_FILE_NAME = "sogou_segment_zh_model";

    public static WordVectorModel loadModel() throws IOException {
        return new WordVectorModel(MODEL_FILE_NAME);
    }

    public static WordVectorModel trainModel() throws IOException {
        //获取绝对地址
        String trainFileName = ResourceUtils.getFile(TRAIN_FILE_NAME).getAbsolutePath();
        String modelFileName = ResourceUtils.getFile(MODEL_FILE_NAME).getAbsolutePath();
        if (!IOUtil.isFileExisted(modelFileName)) {
            if (!IOUtil.isFileExisted(trainFileName)) {
                System.err.println("语料不存在");
                System.exit(1);
            }
            Word2VecTrainer trainerBuilder = new Word2VecTrainer();
            return trainerBuilder.train(trainFileName, modelFileName);
        }

        return loadModel();
    }
}
