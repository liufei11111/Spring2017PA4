package cs276.pa4;

import cs276.pa4.ranklib.TrainFileSaveUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class UtilExtraRankLibPointWise {
  static String[] TFTYPES = {"url","title","body","header","anchor"};
  
  /**
   * Load the training data file
   * @param feature_file_name
   * @return
   * @throws Exception
   */
  public static Map<Query,List<Document>> loadTrainData (String feature_file_name) throws Exception {
    Map<Query, List<Document>> result = new HashMap<Query, List<Document>>();

    File feature_file = new File(feature_file_name);
    if (!feature_file.exists() ) {
      System.err.println("Invalid feature file name: " + feature_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(feature_file));
    String line = null, anchor_text = null;
    Query query = null;
    Document doc = null;
    int numQuery=0; int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = new Query(value);
        numQuery++;
        result.put(query, new ArrayList<Document>());
      } else if (key.equals("url")) {
        doc = new Document();
        doc.url = new String(value);
        result.get(query).add(doc);
        numDoc++;
      } else if (key.equals("title")) {
        doc.title = new String(value);
      } else if (key.equals("header"))
      {
        if (doc.headers == null)
          doc.headers =  new ArrayList<String>();
        doc.headers.add(value);
      } else if (key.equals("body_hits")) {
        if (doc.body_hits == null)
          doc.body_hits = new HashMap<String, List<Integer>>();
        String[] temp = value.split(" ", 2);
        String term = temp[0].trim();
        List<Integer> positions_int;

        if (!doc.body_hits.containsKey(term))
        {
          positions_int = new ArrayList<Integer>();
          doc.body_hits.put(term, positions_int);
        } else
          positions_int = doc.body_hits.get(term);

        String[] positions = temp[1].trim().split(" ");
        for (String position : positions)
          positions_int.add(Integer.parseInt(position));

      } else if (key.equals("body_length"))
        doc.body_length = Integer.parseInt(value);
      else if (key.equals("pagerank"))
        doc.page_rank = Integer.parseInt(value);
      else if (key.equals("anchor_text")) {
        anchor_text = value;
        if (doc.anchors == null)
          doc.anchors = new HashMap<String, Integer>();
      }
      else if (key.equals("stanford_anchor_count"))
        doc.anchors.put(anchor_text, Integer.parseInt(value));      
    }

    reader.close();
    System.err.println("# Signal file " + feature_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);

    return result;
  }

  /**
   * Load the rel data file
   * @param rel_file_name
   * @return
   * @throws IOException
   */
  public static Map<String, Map<String, Double>> loadRelData(String rel_file_name) throws IOException{
    Map<String, Map<String, Double>> result = new HashMap<String, Map<String, Double>>();

    File rel_file = new File(rel_file_name);
    if (!rel_file.exists() ) {
      System.err.println("Invalid feature file name: " + rel_file_name);
      return null;
    }

    BufferedReader reader = new BufferedReader(new FileReader(rel_file));
    String line = null, query = null, url = null;
    int numQuery=0; 
    int numDoc=0;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.split(":", 2);
      String key = tokens[0].trim();
      String value = tokens[1].trim();

      if (key.equals("query")){
        query = value;
        result.put(query, new HashMap<String, Double>());
        numQuery++;
      } else if (key.equals("url")){
        String[] tmps = value.split(" ", 2);
        url = tmps[0].trim();
        double score = Double.parseDouble(tmps[1].trim());
        result.get(query).put(url, score);
        numDoc++;
      }
    }
    reader.close();
    System.err.println("# Rel file " + rel_file_name + ": number of queries=" + numQuery + ", number of documents=" + numDoc);
    
    return result;
  }
  private static boolean isTrain(String refFile){
    return refFile != null;
  }
  /**
   * Load Signal files. Might be helpful in implementing extractTrainFeatures and extractTestFeatures methods
   * @param train_data_file
   * @param idfs
   * @return
   */
  public static Quad<Instances, List<Pair<Query, Document>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
  loadSignalFileLinearRegression(String train_data_file, Map<String,Double> idfs, String refFile,TestFeatures testFeas){

    /* Initial feature vectors */
    Instances X = null;

    /* Build X and Y matrices */
    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    attributes.add(new Attribute("url_w"));
    attributes.add(new Attribute("title_w"));
    attributes.add(new Attribute("body_w"));
    attributes.add(new Attribute("header_w"));
    attributes.add(new Attribute("anchor_w"));
    attributes.add(new Attribute("page_rank"));
    attributes.add(new Attribute("BM_25"));
    attributes.add(new Attribute("min_window"));
    attributes.add(new Attribute("url_sim"));
    attributes.add(new Attribute("title_sim"));
    attributes.add(new Attribute("body_sim"));
    attributes.add(new Attribute("header_sim"));
    attributes.add(new Attribute("anchor_sim"));
//    if (isTrain(refFile)){
    List labels = new ArrayList(9);
    labels.add("0.0");
    labels.add("0.5");
    labels.add("1.0");
    labels.add("1.5");
    labels.add("2.0");
    labels.add("2.5");
    labels.add("3.0");
    labels.add("3.5");
    labels.add("4.0");
      attributes.add(new Attribute("relevance_score",labels));
//    }
    X = new Instances("train_dataset", attributes, 0);
    int numAttributes = X.numAttributes();
    /* Map to record which doc belong to which query: query -> [list of doc] */
    Map<Integer, List<Integer>> index_map = new HashMap<Integer, List<Integer>>();
    int query_counter = 0, doc_counter = 0;
    List<Pair<Query, Document>> queryDocList = new ArrayList<Pair<Query,Document>>();
    Map<Integer,Integer> rowIdToQid = new HashMap<>();
    int rowId = 0;
    try {
      Map<Query,List<Document>> data_map = UtilExtraRankLibPointWise.loadTrainData (train_data_file);
      Map<String, Map<String, Double>> labelMap = null;
      if (isTrain(refFile)){
        labelMap = UtilExtraRankLibPointWise.loadRelData(refFile);
      }
      FeatureExtra feature = new FeatureExtra(idfs, data_map);
      
      /* Add data */
      for (Query query : data_map.keySet()){
        index_map.put(query_counter, new ArrayList<Integer>());
        
        for (Document doc : data_map.get(query)){
          index_map.get(query_counter).add(doc_counter);
          doc_counter ++;
          double[] features = feature.extractMoreFeatures(doc, query,null);
          double[] instance = new double[numAttributes];
          for (int i = 0; i < features.length; ++i) {
            instance[i] = features[i];
          }
          if (isTrain(refFile)){
            instance[numAttributes-1] = labelMap.get(query.query).get(doc.url)/0.5;
          }
          Instance inst = new DenseInstance(1.0, instance);
          X.add(inst);
          queryDocList.add(new Pair<Query, Document>(query, doc));
          rowIdToQid.put(rowId++,query_counter);
        }
        query_counter++;
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
        
    /* Conduct standardization on X */
//    Standardize filter = new Standardize();
    Normalize filter = new Normalize(); filter.setScale(2.0); filter.setTranslation(-1.0); // scale values to [-1, 1]
    Instances new_X = null;   
    try {
      filter.setInputFormat(X); 
      new_X = Filter.useFilter(X, filter);
    } catch (Exception e) {
      e.printStackTrace();
    } 
    new_X.setClass(attributes.get(attributes.size()-1));
    TrainFileSaveUtil.saveFromInstances(new_X,rowIdToQid,Config.saveRankNetTrainFileName);
    testFeas.rowToQueryID = rowIdToQid;
    return new Quad<Instances, List<Pair<Query, Document>>, ArrayList<Attribute>, Map<Integer, List<Integer>>> (new_X, queryDocList, attributes, index_map);
  }


  public static void main(String[] args) {
    try {
      System.out.print(loadRelData(args[0]));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
