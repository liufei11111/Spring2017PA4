package cs276.pa4.embedding;

import cs276.pa4.Config;
import cs276.pa4.Document;
import cs276.pa4.Embedding;
import cs276.pa4.Query;
import cs276.pa4.Util;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;
import org.ejml.simple.SimpleMatrix;

/**
 * Created by feiliu on 5/29/17.
 */
public class GoogleWordVector {
  public double[] getQueryDocEmbeddingSim(Map<String,Double> rawQueryFreq, Map<String,Map<String, Double>> rawDocuFreq){
    Embedding embedding = new Embedding();
    Map<String, SimpleMatrix> wv = embedding.loadEmbeddings(Config.savedEmbeddingFile);
    double[] simScores = new double[5];
    simScores[0]=getSimScore(rawQueryFreq,rawDocuFreq.get("url"),wv);//title, header, anchor, body
    simScores[1]=getSimScore(rawQueryFreq,rawDocuFreq.get("title"),wv);
    simScores[2]=getSimScore(rawQueryFreq,rawDocuFreq.get("header"),wv);
    simScores[3]=getSimScore(rawQueryFreq,rawDocuFreq.get("anchor"),wv);
    simScores[4]=getSimScore(rawQueryFreq,rawDocuFreq.get("body"),wv);
    return simScores;
  }

  private double getSimScore(Map<String,Double> rawQueryFreq, Map<String, Double> rawDocuFre, Map<String, SimpleMatrix> wv) {
    SimpleMatrix queryAvg = null;
    double count = 0.0;
    for (String qW : rawQueryFreq.keySet()){
      SimpleMatrix qwValue = wv.get(qW);
      if (qwValue == null){
        double[][] var1 = new double[300][1];
        qwValue = new SimpleMatrix(var1);
      }else{
        qwValue = qwValue.divide(1.0/rawDocuFre.get(qW));
        count+=rawDocuFre.get(qW);
      }
      if (queryAvg == null){
        queryAvg = qwValue;
      }else{
        queryAvg=queryAvg.plus(qwValue);
      }
    }
    queryAvg.divide(count);
    SimpleMatrix DocAvg = null;
    count = 0.0;
    for (String qW : rawDocuFre.keySet()){
      SimpleMatrix qwValue = wv.get(qW);
      if (qwValue == null){
        double[][] var1 = new double[300][1];
        qwValue = new SimpleMatrix(var1);
      }else{
        qwValue = qwValue.divide(1.0/rawDocuFre.get(qW));
        count+=rawDocuFre.get(qW);
      }
      if (DocAvg == null){
        DocAvg = qwValue;
      }else{
        DocAvg=DocAvg.plus(qwValue);
      }
    }
    DocAvg.divide(count);
    double score = 0.0;
    for (int i=0;i<Config.embeddingSize;++i){
      score +=queryAvg.get(i)*DocAvg.get(i);
    }
    return score;
  }
}
