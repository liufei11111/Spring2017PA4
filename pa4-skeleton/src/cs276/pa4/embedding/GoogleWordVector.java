package cs276.pa4.embedding;

import cs276.pa4.Util;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Map;

/**
 * Created by feiliu on 5/29/17.
 */
public class GoogleWordVector {
  public static void main(String[] args) throws IOException {
    Map<String,Double> idfs = Util.loadDFs("./pa4-data/idfs");
    FileWriter fw = new FileWriter(new File("output/vocabulary.txt"));
    for (String str : idfs.keySet()){
      fw.write(str+"\n");
    }
    fw.close();
  }
}
