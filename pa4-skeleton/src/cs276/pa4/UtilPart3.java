package cs276.pa4;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

public class UtilPart3 {

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

  public static Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
  loadSignalFileLinearSVM(String train_data_file, Map<String,Double> idfs, String refFile){

    /* Initial feature vectors */
    Instances X = null;

    /* Build X and Y matrices */
    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    attributes.add(new Attribute("url_w"));
    if (!Config.isRazarEnabledPart3){
      attributes.add(new Attribute("title_w"));
    }

    attributes.add(new Attribute("body_w"));
    attributes.add(new Attribute("header_w"));
    attributes.add(new Attribute("anchor_w"));
    attributes.add(new Attribute("page_rank"));
    attributes.add(new Attribute("BM_25"));
    attributes.add(new Attribute("min_window"));
//    if (isTrain(refFile)){
    List labels = new ArrayList(2);
    labels.add("positive");
    labels.add("negative");

    attributes.add(new Attribute("relevance_score", labels));
//    }
    X = new Instances("train_dataset", attributes, 0);
    int numAttributes = X.numAttributes();
    /* Map to record which doc belong to which query: query -> [list of doc] */
    Map<Integer, List<Integer>> index_map = new HashMap<Integer, List<Integer>>();
    int query_counter = 0, doc_counter = 0;
    List<Pair<Query, Pair<Document,Document>>> queryDocList = new ArrayList<Pair<Query,Pair<Document,Document>>>();
    try {
      Map<Query,List<Document>> data_map = Util.loadTrainData (train_data_file);
      Map<String, Map<String, Double>> labelMap = null;
      if (isTrain(refFile)){
        labelMap = Util.loadRelData(refFile);
      }
      Feature feature = new Feature(idfs, data_map);
      /* Add data */
      for (Query query : data_map.keySet()){
        index_map.put(query_counter, new ArrayList<Integer>());
        List<Document> docs = data_map.get(query);
        if (isTrain(refFile)){
          List<Pair<Document,Double>> tempList = new ArrayList<>();
          for(Document doc : docs){
            tempList.add(new Pair(doc,labelMap.get(query.query).get(doc.url)));
          }
          Collections.sort(tempList, new Comparator<Pair<Document, Double>>() {
            @Override
            public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
              return o2.getSecond().compareTo(o1.getSecond());
            }
          });
          List<Document> sortedDocs = new ArrayList<>();
          for (Pair<Document,Double> sortedDoc : tempList){
            sortedDocs.add(sortedDoc.getFirst());
          }
          docs = sortedDocs;
        }
        boolean isOdd = true;
        for (int i=0;i<docs.size()-1;++i){
          Document doc = docs.get(i);

          index_map.get(query_counter).add(doc_counter);
          doc_counter ++;

          for (int j = i+1;j<docs.size();++j){
            double[] features = feature.extractMoreFeatures(doc, query,null);
            Document doc2 = docs.get(j);
            double[] features2 = feature.extractMoreFeatures(doc2, query,null);
            double[] instance = new double[numAttributes];
            for (int k = 0; k < features.length; ++k) {
              instance[k] = features[k]-features2[k];
            }

            if (isTrain(refFile)){
              if (isOdd){
                for (int k = 0; k < features.length; ++k) {
                  instance[k] = -instance[k];
                }
                instance[numAttributes-1] = 1.0;
              }else{
                instance[numAttributes-1] = 0.0;
              }
              isOdd = !isOdd;

            }
            Instance inst = new DenseInstance(1.0, instance);
            X.add(inst);
            queryDocList.add(new Pair(query,  new Pair(doc,doc2)));
          }




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
    return new Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>> (new_X, queryDocList, attributes, index_map);
  }
  public static Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
  loadSignalFileLinearSVMTraining(String train_data_file, Map<String,Double> idfs, String refFile){

    /* Initial feature vectors */
    Instances X = null;

    /* Build X and Y matrices */
    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
    attributes.add(new Attribute("url_w"));
    if (!Config.isRazarEnabledPart3){
      attributes.add(new Attribute("title_w"));
    }
    attributes.add(new Attribute("body_w"));
    attributes.add(new Attribute("header_w"));
    attributes.add(new Attribute("anchor_w"));
    attributes.add(new Attribute("page_rank"));
    attributes.add(new Attribute("BM_25"));
    attributes.add(new Attribute("min_window"));
//    if (isTrain(refFile)){
    List labels = new ArrayList(2);
    labels.add("positive");
    labels.add("negative");

    attributes.add(new Attribute("relevance_score", labels));
//    }
    X = new Instances("train_dataset", attributes, 0);
    int numAttributes = X.numAttributes();
    /* Map to record which doc belong to which query: query -> [list of doc] */
    Map<Integer, List<Integer>> index_map = new HashMap<Integer, List<Integer>>();
    int query_counter = 0, doc_counter = 0;
    List<Pair<Query, Pair<Document,Document>>> queryDocList = new ArrayList<Pair<Query,Pair<Document,Document>>>();
    try {
      Map<Query,List<Document>> data_map = Util.loadTrainData (train_data_file);
      Map<String, Map<String, Double>> labelMap = null;
      if (isTrain(refFile)){
        labelMap = Util.loadRelData(refFile);
      }
      Feature feature = new Feature(idfs, data_map);
      /* Add data */
      for (Query query : data_map.keySet()){
        index_map.put(query_counter, new ArrayList<Integer>());
        List<Document> docs = data_map.get(query);
        List<Pair<Document,Double>> docPairs = null;
        if (isTrain(refFile)){
          docPairs = new ArrayList<>();
          for(Document doc : docs){
            docPairs.add(new Pair(doc,labelMap.get(query.query).get(doc.url)));
          }
          Collections.sort(docPairs, new Comparator<Pair<Document, Double>>() {
            @Override
            public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
              return o2.getSecond().compareTo(o1.getSecond());
            }
          });
          List<Document> sortedDocs = new ArrayList<>();
          for (Pair<Document,Double> sortedDoc : docPairs){
            sortedDocs.add(sortedDoc.getFirst());
          }

        }
        boolean isOdd = true;
        for (int i=0;i<docPairs.size()-1;++i){
          Pair<Document,Double> doc = docPairs.get(i);

          index_map.get(query_counter).add(doc_counter);
          doc_counter ++;

          for (int j = i+1;j<docPairs.size();++j){
            Pair<Document,Double> doc2 = docPairs.get(j);
            if (doc2.getSecond() == doc.getSecond()){
              continue;// do not add information if they are equal
            }

            double[] features = feature.extractMoreFeatures(doc.getFirst(), query,null);

            double[] features2 = feature.extractMoreFeatures(doc2.getFirst(), query, null);
            double[] instance = new double[numAttributes];
            for (int k = 0; k < features.length; ++k) {
              instance[k] = features[k]-features2[k];
            }

            if (isTrain(refFile)){
              if (isOdd){
                for (int k = 0; k < features.length; ++k) {
                  instance[k] = -instance[k];
                }
                instance[numAttributes-1] = 1.0;
              }else{
                instance[numAttributes-1] = 0.0;
              }
              isOdd = !isOdd;

            }
            Instance inst = new DenseInstance(1.0, instance);
            X.add(inst);
            queryDocList.add(new Pair(query,  new Pair(doc,doc2)));
          }




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
    return new Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>> (new_X, queryDocList, attributes, index_map);
  }
  /**
   * Load document-frequencies. Could also be used to load the idfs file supplied with PA4 data.
   * @param idfFile
   * @return
   */
  public static Map<String,Double> loadDFs(String idfFile) {
    Map<String,Double> termDocCount = null;
    try {
      FileInputStream fis = new FileInputStream(idfFile);
      ObjectInputStream ois = new ObjectInputStream(fis);
      termDocCount = (HashMap<String,Double>) ois.readObject();
      ois.close();
      fis.close();
    }
    catch(IOException | ClassNotFoundException ioe) {
      ioe.printStackTrace();
      return null;
    }
    return termDocCount;
  }


  /**
   * Extract various kinds of term frequencies eg. (url, title, body, header, and anchor)
   * @param d
   * @param q
   * @return
   */
  public static Map<String,Map<String, Double>> getDocTermFreqs(Document d, Query q)
  {
    String url = d.url;

    //map from tf type -> queryWord -> score
    Map<String,Map<String, Double>> tfs = new HashMap<String,Map<String, Double>>();

    ////////////////////Initialization/////////////////////

    //initialize tfs
    for (String type : Util.TFTYPES)
      tfs.put(type, new HashMap<String,Double>());

    //initialize tfs for querywords
    for (String queryWord : q.queryWords)
      for (String tfType : tfs.keySet())
        tfs.get(tfType).put(queryWord, 0.0);

    //tokenize url
    ArrayList<String> urlTokens = Parser.parseUrlString(url);

    //tokenize title
    List<String> titleTokens = Parser.parseTitle(d.title);

    //tokenize headers if exist
    List<String> headerTokens = Parser.parseHeaders(d.headers);

    //tokenize anchors if exist
    Map<String,Integer> anchorCountMap = Parser.parseAnchors(d.anchors);

    //tokenize body_hits if exists
    Map<String,Integer> bodyCountMap = Parser.parseBody(d.body_hits);

    ////////////////////////////////////////////////////////

    //////////handle counts//////

    //loop through query terms increasing relevant tfs
    for (String queryWord : q.queryWords)
    {
      //url
      for (String urlToken : urlTokens)
      {

        if (queryWord.equals(urlToken))
        {
          double oldVal = tfs.get("url").get(queryWord);
          tfs.get("url").put(queryWord, oldVal+1);
        }
      }

      //title
      for (String titleToken : titleTokens)
      {
        if (queryWord.equals(titleToken))
        {
          double oldVal = tfs.get("title").get(queryWord);
          tfs.get("title").put(queryWord, oldVal+1);
        }
      }

      //headers --if size is 0, just skip over
      for (String headerToken : headerTokens)
      {
        if (queryWord.equals(headerToken))
        {
          double oldVal = tfs.get("header").get(queryWord);
          tfs.get("header").put(queryWord, oldVal+1);
        }
      }

      //anchors --if none, skip over
      for (String anchorToken : anchorCountMap.keySet())
      {
        if (queryWord.equals(anchorToken))
        {
          double oldVal = tfs.get("anchor").get(queryWord);
          tfs.get("anchor").put(queryWord, oldVal+anchorCountMap.get(anchorToken));
        }
      }

      //body_hits--if none, skip over
      for (String bodyHit : bodyCountMap.keySet())
      {
        if (queryWord.equals(bodyHit))
        {
          double oldVal = tfs.get("body").get(queryWord);
          tfs.get("body").put(queryWord, oldVal+bodyCountMap.get(bodyHit));
        }
      }

    }
    return tfs;
  }

  public static void main(String[] args) {
    try {
      System.out.print(loadRelData(args[0]));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
