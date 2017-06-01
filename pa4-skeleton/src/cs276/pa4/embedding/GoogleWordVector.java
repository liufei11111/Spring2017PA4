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
  static Map<String, SimpleMatrix> wv =null;
  public static Map<String, SimpleMatrix> getEmbedding(){
    if (wv == null){
      Embedding embedding = new Embedding();
      wv = embedding.loadEmbeddings(Config.savedEmbeddingFile);
    }
    return wv;
  }
  public static double[] getQueryDocEmbeddingSim(Map<String,Double> rawQueryFreq, Map<String,Map<String, Double>> rawDocuFreq){
    Map<String, SimpleMatrix> wv = GoogleWordVector.getEmbedding();
    double[] simScores = new double[5];
    simScores[0]=rawDocuFreq.get("url").size()>0?getSimScore(rawQueryFreq,rawDocuFreq.get("url"),wv):0.0;//title, header, anchor, body
    simScores[1]=rawDocuFreq.get("title").size()>0?getSimScore(rawQueryFreq,rawDocuFreq.get("title"),wv):0.0;
    simScores[2]=rawDocuFreq.get("header").size()>0?getSimScore(rawQueryFreq,rawDocuFreq.get("header"),wv):0.0;
    simScores[3]=rawDocuFreq.get("anchor").size()>0?getSimScore(rawQueryFreq,rawDocuFreq.get("anchor"),wv):0.0;
    simScores[4]=rawDocuFreq.get("body").size()>0?getSimScore(rawQueryFreq,rawDocuFreq.get("body"),wv):0.0;
    return simScores;
  }
  private static SimpleMatrix getSimVector(Map<String,Double> rawQueryFreq,Map<String, SimpleMatrix> wv){
//    System.out.println("wv: size: "+wv.size());
    SimpleMatrix  queryAvg = null;
    double count = 0.0;
    for (String qW : rawQueryFreq.keySet()){
      SimpleMatrix qwValue = wv.get(qW);
      if (qwValue == null){
        double[][] var1 = new double[300][1];
        qwValue = new SimpleMatrix(var1);
      }else{
        qwValue = qwValue.divide(1.0/rawQueryFreq.get(qW));
        count+=rawQueryFreq.get(qW);
      }
      if (queryAvg == null){
        queryAvg = qwValue;
      }else{
        queryAvg=queryAvg.plus(qwValue);
      }
    }
    queryAvg.divide(count);
    return queryAvg;
  }
  private static  double getSimScore(Map<String,Double> rawQueryFreq, Map<String, Double> rawDocuFre, Map<String, SimpleMatrix> wv) {
//    System.out.println("rawDocuFre: size: "+rawDocuFre.size());
    SimpleMatrix queryAvg =getSimVector( rawQueryFreq, wv);

    SimpleMatrix DocAvg = getSimVector( rawDocuFre, wv);

    double score = 0.0;
    DocAvg.minus(queryAvg);
    for (int i=0;i<Config.embeddingSize;++i){
      score +=DocAvg.get(i)*DocAvg.get(i);
    }
    return score;
  }
}
