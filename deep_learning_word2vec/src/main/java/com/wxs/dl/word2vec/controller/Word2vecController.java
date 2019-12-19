package com.wxs.dl.word2vec.controller;

import com.medallia.word2vec.Searcher;
import com.wxs.dl.word2vec.service.Word2vecService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * @Author: yoyo
 * @Description:
 * @Date: Created in 2019/12/19 15:49
 */
@RestController
@RequestMapping("/w2v")
public class Word2vecController {

    @Autowired
    private Word2vecService word2vecService;


    @GetMapping("/model")
    public Object getModel() {
        return word2vecService.train();
    }

    @GetMapping("/rawVector")
    public Object getRawVector(String name) throws Searcher.UnknownWordException {
        return word2vecService.train().forSearch().getRawVector(name);
    }

    @GetMapping("/vocab")
    public Object getVocab() {
        return word2vecService.train().getVocab();
    }

    @GetMapping("/matches")
    public Object getMatches(@RequestParam(defaultValue = "") String name) throws Searcher.UnknownWordException {
        return word2vecService.train().forSearch().getMatches(name, 10);
    }

}
