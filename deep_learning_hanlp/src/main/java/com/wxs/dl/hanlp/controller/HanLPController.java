package com.wxs.dl.hanlp.controller;

import com.hankcs.hanlp.mining.word2vec.WordVectorModel;
import com.wxs.dl.hanlp.service.HanLPWord2VecService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

/**
 * @Author: yoyo
 * @Description:
 * @Date: Created in 2019/12/19 15:49
 */
@RestController
@RequestMapping("/hanlp")
public class HanLPController {

    @GetMapping("/model")
    public WordVectorModel getModel() throws IOException {
        return HanLPWord2VecService.trainModel();
    }


    @GetMapping("/similarity")
    public float getSimilarity(@RequestParam(defaultValue = "上海") String what, @RequestParam(defaultValue = "广州") String with) throws IOException {
        return HanLPWord2VecService.trainModel().similarity(what, with);
    }

}
