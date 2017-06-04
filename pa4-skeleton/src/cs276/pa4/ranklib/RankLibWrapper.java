package cs276.pa4.ranklib;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.learning.boosting.AdaRank;
import ciir.umass.edu.learning.boosting.RankBoost;
import ciir.umass.edu.learning.neuralnet.ListNet;
import ciir.umass.edu.learning.neuralnet.RankNet;
import cs276.pa4.Config;
import cs276.pa4.Document;
import java.util.Map;
import weka.core.Instance;

/**
 * Created by feiliu on 6/3/17.
 */
public class RankLibWrapper {

  public static RankBoost loadModel(){
    RankBoost rb = new RankBoost();
    rb.load(Config.saveRankNetModelFileName);
    return rb;
  }
  public static RankNet loadModelRankNet(){
    RankNet rb = new RankNet();
    rb.load(Config.saveRankNetModelFileName);
    return rb;
  }
  public static AdaRank loadModelAdaRank(){
    AdaRank rb = new AdaRank();
    rb.load(Config.saveRankNetModelFileName);
    return rb;
  }
  public static ListNet loadModelListNet(){
    ListNet rb = new ListNet();
    rb.load(Config.saveRankNetModelFileName);
    return rb;
  }
  public static DataPoint convertToDataPoint(int rawIndex, Instance inst,Map<Integer, Integer> rowToQueryID) {
    StringBuilder sb = new StringBuilder();
    sb.append((int)inst.value(inst.numAttributes()-1));
    sb.append(" qid:");
    sb.append(rowToQueryID.get(rawIndex)+" ");
    for (int k=0;k<inst.numAttributes()-1;++k){
      sb.append((k+1)+":"+inst.value(k));
      if (k!= inst.numAttributes()-2){
        sb.append(" ");
      }
    }
    DataPoint dp = new DataPoint(sb.toString());
    return dp;
  }
}
