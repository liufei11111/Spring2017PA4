package cs276.pa4.ranklib;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import cs276.pa4.Config;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by feiliu on 6/3/17.
 */
public class TrainFileSaveUtil {

  public static void saveFromInstances(Instances new_x, Map<Integer, Integer> map, String saveFileName) {

    BufferedWriter fw = null;
    try {
      fw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(saveFileName))));
      StringBuilder sb = new StringBuilder();
      for (int i=0;i<new_x.size();++i){
        Instance inst = new_x.get(i);
//        for (int j=0;j<inst.numAttributes();++j){

          sb.append((int)inst.value(inst.numAttributes()-1));
          sb.append(" qid:");
          sb.append(map.get(i)+" ");
          for (int k=0;k<inst.numAttributes()-1;++k){
            sb.append((k+1)+":"+inst.value(k));
            if (k!= inst.numAttributes()-2){
              sb.append(" ");
            }
          }

//        }

        fw.write(sb.toString());
        fw.newLine();
        sb.setLength(0);
      }


      fw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }
}
