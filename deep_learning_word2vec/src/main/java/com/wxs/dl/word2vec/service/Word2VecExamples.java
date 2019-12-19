package com.wxs.dl.word2vec.service;

import com.google.common.collect.Lists;
import com.medallia.word2vec.Searcher;
import com.medallia.word2vec.Word2VecModel;
import com.medallia.word2vec.Word2VecTrainerBuilder;
import com.medallia.word2vec.neuralnetwork.NeuralNetworkType;
import com.medallia.word2vec.thrift.Word2VecModelThrift;
import com.medallia.word2vec.util.*;
import org.apache.commons.io.FileUtils;
import org.apache.commons.logging.Log;
import org.apache.thrift.TException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * @Author: yoyo
 * @Description:
 * @Date: Created in 2019/12/19 16:34
 */
public class Word2VecExamples {
    private static final Log LOG = AutoLog.getLog();

    /**
     * Runs the example
     */
    public static void main(String[] args) throws IOException, TException, Searcher.UnknownWordException, InterruptedException {
        demoWord();
    }

    /**
     * Trains a model and allows user to find similar words
     * demo-word.sh example from the open source C implementation
     */
    public static void demoWord() throws IOException, TException, InterruptedException, Searcher.UnknownWordException {
        //File f = new File("text8");
        //File f = new File("D:\\git\\github\\ai\\deep_learning_java\\deep_learning_word2vec\\src\\main\\resources\\data\\text8");
        File f = new File("D:\\git\\github\\ai\\deep_learning_java\\deep_learning_word2vec\\src\\main\\resources\\data\\text8_mini");
        if (!f.exists())
            throw new IllegalStateException("Please download and unzip the text8 example from http://mattmahoney.net/dc/text8.zip");
        List<String> read = Common.readToList(f);
        List<List<String>> partitioned = Lists.transform(read, new com.google.common.base.Function<String, List<String>>() {
            @Override
            public List<String> apply(String input) {
                return Arrays.asList(input.split(" "));
            }
        });

        Word2VecModel model = Word2VecModel.trainer()
                .setMinVocabFrequency(5)
                .useNumThreads(20)
                .setWindowSize(8)
                .type(NeuralNetworkType.CBOW)
                .setLayerSize(200)
                .useNegativeSamples(25)
                .setDownSamplingRate(1e-4)
                .setNumIterations(5)
                .setListener(new Word2VecTrainerBuilder.TrainingProgressListener() {
                    @Override
                    public void update(Word2VecTrainerBuilder.TrainingProgressListener.Stage stage, double progress) {
                        System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
                    }
                })
                .train(partitioned);

        // Writes model to a thrift file
        try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
            FileUtils.writeStringToFile(new File("text8.model"), ThriftUtils.serializeJson(model.toThrift()));
        }

        // Alternatively, you can write the model to a bin file that's compatible with the C
        // implementation.
        try (final OutputStream os = Files.newOutputStream(Paths.get("text8.bin"))) {
            model.toBinFile(os);
        }

        interact(model.forSearch());
    }

    /**
     * Loads a model and allows user to find similar words
     */
    public static void loadModel() throws IOException, TException, Searcher.UnknownWordException {
        final Word2VecModel model;
        try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Loading model")) {
            String json = Common.readFileToString(new File("text8.model"));
            model = Word2VecModel.fromThrift(ThriftUtils.deserializeJson(new Word2VecModelThrift(), json));
        }
        interact(model.forSearch());
    }

    /**
     * Example using Skip-Gram model
     */
    public static void skipGram() throws IOException, TException, InterruptedException, Searcher.UnknownWordException {
        List<String> read = Common.readToList(new File("sents.cleaned.word2vec.txt"));
        List<List<String>> partitioned = Lists.transform(read, new com.google.common.base.Function<String, List<String>>() {
            @Override
            public List<String> apply(String input) {
                return Arrays.asList(input.split(" "));
            }
        });

        Word2VecModel model = Word2VecModel.trainer()
                .setMinVocabFrequency(100)
                .useNumThreads(20)
                .setWindowSize(7)
                .type(NeuralNetworkType.SKIP_GRAM)
                .useHierarchicalSoftmax()
                .setLayerSize(300)
                .useNegativeSamples(0)
                .setDownSamplingRate(1e-3)
                .setNumIterations(5)
                .setListener(new Word2VecTrainerBuilder.TrainingProgressListener() {
                    @Override
                    public void update(Word2VecTrainerBuilder.TrainingProgressListener.Stage stage, double progress) {
                        System.out.println(String.format("%s is %.2f%% complete", Format.formatEnum(stage), progress * 100));
                    }
                })
                .train(partitioned);

        try (ProfilingTimer timer = ProfilingTimer.create(LOG, "Writing output to file")) {
            FileUtils.writeStringToFile(new File("300layer.20threads.5iter.model"), ThriftUtils.serializeJson(model.toThrift()));
        }

        interact(model.forSearch());
    }

    private static void interact(Searcher searcher) throws IOException, Searcher.UnknownWordException {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.print("Enter word or sentence (EXIT to break): ");
                String word = br.readLine();
                if (word.equals("EXIT")) {
                    break;
                }
                List<Searcher.Match> matches = searcher.getMatches(word, 20);
                System.out.println(Strings.joinObjects("\n", matches));
            }
        }
    }
}
