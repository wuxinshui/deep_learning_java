package com.wxs.dl.word2vec.service;

import com.google.common.collect.Lists;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.util.Format;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.Arrays;
import java.util.List;

/**
 * @Author: yoyo
 * @Description:
 * @Date: Created in 2019/12/19 15:47
 */
@Service
@Slf4j
public class Word2vecService {

    public Word2VecModel train() {
        try {
            List<String> data = List.of();
            List list = Lists.transform(data, var11 -> Arrays.asList(var11.split(" ")));
            Word2VecModel word2VecModel = Word2VecModel.trainer().setMinVocabFrequency(5).useNumThreads(20).setWindowSize(8).type(NeuralNetworkType.CBOW).setLayerSize(200).useNegativeSamples(25).setDownSamplingRate(1.0E-4D).setNumIterations(5).setListener((var1, var2) -> System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(var1), var2 * 100.0D))).train(list);
            return word2VecModel;
        } catch (InterruptedException e) {
            log.error("exception:{}", e);
            return null;
        }
    }
}
