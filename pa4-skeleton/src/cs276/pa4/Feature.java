package cs276.pa4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.Set;
import java.util.HashSet;

public class Feature {
  
  public static boolean isSublinearScaling = true;
  private Parser parser = new Parser();
  double smoothingBodyLength = 800;

  Map<String,Double> idfs;

  // Various types of term frequencies that you will need
  String[] TFTYPES = {"url","title","body","header","anchor"};

  public double urlweight = 0.23795549267542235;
  public double titleweight  = 0.09096936314246667;
  public double bodyweight = 0.018406977715679264;
  public double headerweight = 0.25588594518954305;
  public double anchorweight = 0.39678222127688867;

  Map<String, Double> Wf;

  // BM25-specific weights
  public double burl = 0.75;
  public double btitle = 0.75;
  public double bheader = 0.75;
  public double bbody = 0.75;
  public double banchor = 0.75;

  Map<String, Double> Bf;

  public double k1 = 2.48;

//query -> url -> document
 Map<Query,List<Document>> queryDict;

 Set<Document> docs;

 // Document -> field -> length
 Map<Document,Map<String,Double>> lengths;

 // field name -> average length
 Map<String,Double> avgLengths;

 public double Bo = 1.554;
    
  // If you would like to use additional features, you can declare necessary variables here
  /*
   * @TODO: Your code here
   */
  
  public Feature(Map<String,Double> idfs, Map<Query,List<Document>> dataMap){
    this.idfs = idfs;
    this.queryDict = dataMap;
    this.calcAverageLengths();
  }

//  public Feature(Map<String,Double> idfs){
//    this.idfs = idfs;
//  }
  
  private void calcAverageLengths() {
    lengths = new HashMap<Document,Map<String,Double>>();
    avgLengths = new HashMap<String,Double>();
    docs = new HashSet<Document>();
    for (Query query : queryDict.keySet()) {
      List<Document> urlDocs = queryDict.get(query);
      docs.addAll(urlDocs);
    }
    Bf = new HashMap<String, Double>();
    Bf.put("url", burl);
    Bf.put("title", btitle);
    Bf.put("body", bheader);
    Bf.put("header", bbody);
    Bf.put("anchor", banchor);
    Wf = new HashMap<String, Double>();
    Wf.put("url", urlweight);
    Wf.put("title", titleweight);
    Wf.put("body", bodyweight);
    Wf.put("header", headerweight);
    Wf.put("anchor", anchorweight);

    for (Query query : queryDict.keySet()) {
      List<Document> urlDocs = queryDict.get(query);
      for (Document doc : urlDocs) {
        Map<String,Double> fieldLength = new HashMap<String, Double>();
        for (String tfType : this.TFTYPES) {
          double len = 0;
          switch(tfType) {
            case "url":
              len = getUrlLength(doc.url);
              break;
            case "title":
              len = getStringLength(doc.title);
              break;
            case "body":
              len = doc.body_length;
              break;
            case "header":
              len = getHeaderLength(doc.headers);
              break;
            case "anchor":
              len = getAnchorLength(doc.anchors);
              break;
          }
          fieldLength.put(tfType, len);
        }
        lengths.put(doc, fieldLength);
      }
    }

    double docNum = docs.size();
    for (String tfType : this.TFTYPES) {
      double totalLen = 0;
      for (Document doc : docs) {
        Map<String, Double> fieldLength = lengths.get(doc);
        totalLen = totalLen + fieldLength.get(tfType);
      }
      avgLengths.put(tfType, totalLen / docNum);
    }
  }

  private double getUrlLength(String s) {
    if (s == null) {
      return 0.0;
    }
    String[] t = s.split("[^A-Za-z0-9]+", -1);
    double l = 0;
    for (String k : t) {
      if (!k.isEmpty()) {
        l++;
      }
    }
    return l;
  }

  private double getStringLength(String s) {
    if (s == null) {
      return 0.0;
    }
    String[] t = s.split(" ", -1);
    double l = 0;
    for (String k : t) {
      if (!k.isEmpty()) {
        l++;
      }
    }
    return l;
  }

  private double getHeaderLength(List<String> sg) {
    if (sg == null) {
      return 0.0;
    }
    double l = 0;
    for (String s : sg) {
      l = l + getStringLength(s);
    }
    return l;
  }

  private double getAnchorLength(Map<String, Integer> sm) {
    if (sm == null) {
      return 0.0;
    }
    double l = 0.0;
    for (String s : sm.keySet()) {
      l = l + getStringLength(s) * sm.get(s);
    }
    return l;
  }

  private double getBm25Score(Map<String,Map<String, Double>> tfs, Query q, Map<String,Double> tfQuery,Document d) {
    double score = 0.0;
    for (String term : tfQuery.keySet()) {
      double termScore = 0.0;
      for (String tfType : tfs.keySet()) {
        Map<String, Double> termFreqHolder = tfs.get(tfType);
        double freq = 0.0;
        if (termFreqHolder.containsKey(term)) {
          freq = termFreqHolder.get(term);
        }
        termScore = termScore + freq * Wf.get(tfType);
      }
      double idfsTerm = 0.0;
      if (!idfs.containsKey(term)) {
        idfsTerm = Math.log(98998.0);
      } else {
        idfsTerm = idfs.get(term);
      }
      score = score + (termScore / (k1 + termScore) * idfsTerm) * tfQuery.get(term);
    }
    return score;
  }

  private int getSmallestFieldWindow(Map<String,Double> tfQuery, String s, String reg) {
    if (s == null || s.isEmpty()) {
      return Integer.MAX_VALUE;
    }
    int needToSatisfy = tfQuery.size();
    Map<String,Double> sQuery = new HashMap<String, Double>();
    String[] t = s.split(reg, -1);
    int sl = 0;
    for (String k : t) {
      if (!k.isEmpty()) {
        sl++;
      }
    }
    int smallestFieldWindow = Integer.MAX_VALUE;
    int satisfied = 0;
    int index1 = 0;
    int index2 = 0;
    int rindex1 = 0;
    int rindex2 = 0;
    while (t[rindex1].isEmpty()) {
      rindex1++;
      continue;
    }
    while (t[rindex2].isEmpty()) {
      rindex2++;
      continue;
    }
    while(index2 < sl) {
      String sIndex2 = t[rindex2];
      double SOccurence = 0.0;
      if (sQuery.containsKey(sIndex2)) {
        SOccurence = sQuery.get(sIndex2);
        sQuery.put(sIndex2, SOccurence + 1);
      } else {
        sQuery.put(sIndex2, 1.0);
      }
      double TfOccurence = 0.0;
      if (tfQuery.containsKey(sIndex2)) {
        TfOccurence = tfQuery.get(sIndex2);
      }
      if (SOccurence == TfOccurence - 1) {
        satisfied++;
      }
      index2++;
      rindex2++;
      while (index2 < sl &&t[rindex2].isEmpty()) {
        rindex2++;
        continue;
      }
      if (satisfied == needToSatisfy) {
        while(true) {
          String sIndex1 = t[rindex1];
          double oldTfPreOccurence = 0.0;
          if (tfQuery.containsKey(sIndex1)) {
            oldTfPreOccurence = tfQuery.get(sIndex1);
          }
          double oldSPreOccurence = sQuery.get(sIndex1);
          if (oldSPreOccurence > oldTfPreOccurence) {
            if (oldSPreOccurence == 1) {
              sQuery.remove(sIndex1);
            } else {
              sQuery.put(sIndex1, oldSPreOccurence - 1);
            }
            index1++;
            rindex1++;
            while (index1 < sl && t[rindex1].isEmpty()) {
              rindex1++;
              continue;
            }
          } else {
            break;
          }
        }
        if (index2 - index1 < smallestFieldWindow) {
          smallestFieldWindow = index2 - index1;
        }
      }
    }
    return smallestFieldWindow;
  }

  private int getSmallestFieldWindow(Map<String,Double> tfQuery, Map<String, List<Integer>> bodyHits) {
    if (bodyHits == null) {
      return Integer.MAX_VALUE;
    }
    int needToSatisfy = tfQuery.size();
    Map<String,Double> sQuery = new HashMap<String, Double>();
    Map<Integer, String> t = new TreeMap<Integer, String>();
    List<Integer> arrT = new ArrayList<Integer>();
    for (String k : bodyHits.keySet()) {
      for (Integer pos : bodyHits.get(k)) {
        t.put(pos, k);
      }
    }
    for (Integer k : t.keySet()) {
      arrT.add(k);
    }
    int sl = arrT.size();
    int smallestFieldWindow = Integer.MAX_VALUE;
    int satisfied = 0;
    int index1 = 0;
    int index2 = 0;
    while(index2 < sl) {
      Integer pos2 = arrT.get(index2);
      String sIndex2 = t.get(pos2);
      double SOccurence = 0.0;
      if (sQuery.containsKey(sIndex2)) {
        SOccurence = sQuery.get(sIndex2);
      }
      sQuery.put(sIndex2, SOccurence + 1);
      double TfOccurence = 0.0;
      if (tfQuery.containsKey(sIndex2)) {
        TfOccurence = tfQuery.get(sIndex2);
      }
      if (SOccurence == TfOccurence - 1) {
        satisfied++;
      }
      index2++;
      if (satisfied == needToSatisfy) {
        while(true) {
          Integer pos1 = arrT.get(index1);
          String sIndex1 = t.get(pos1);
          double oldTfPreOccurence = 0.0;
          if (tfQuery.containsKey(sIndex1)) {
            oldTfPreOccurence = tfQuery.get(sIndex1);
          }
          double oldSPreOccurence = sQuery.get(sIndex1);
          if (oldSPreOccurence > oldTfPreOccurence) {
            if (oldSPreOccurence == 1) {
              sQuery.remove(sIndex1);
            } else {
              sQuery.put(sIndex1, oldSPreOccurence - 1);
            }
            index1++;
          } else {
            break;
          }
        }
        if (arrT.get(index2 - 1) - arrT.get(index1) + 1 < smallestFieldWindow) {
          smallestFieldWindow = arrT.get(index2 - 1) - arrT.get(index1) + 1;
        }
      }
    }
    return smallestFieldWindow;
  }

  private int getSmallestFieldWindow(Map<String,Double> tfQuery, Document d, String tfType) {
    int smallestFieldWindow = Integer.MAX_VALUE;
    int tempSmallestFieldWindow = Integer.MAX_VALUE;
    switch(tfType) {
    case "url":
      smallestFieldWindow = getSmallestFieldWindow(tfQuery, d.url, "[^A-Za-z0-9]+");
      break;
    case "title":
      smallestFieldWindow = getSmallestFieldWindow(tfQuery, d.title, " ");
      break;
    case "body":
      if (d.body_hits == null || tfQuery.size() != d.body_hits.size()) {
        return Integer.MAX_VALUE;
      }
      smallestFieldWindow = getSmallestFieldWindow(tfQuery, d.body_hits);
      break;
    case "header":
      if (d.headers == null) {
        return Integer.MAX_VALUE;
      }
      for (String header : d.headers) {
        tempSmallestFieldWindow = getSmallestFieldWindow(tfQuery, header, " ");
        if (tempSmallestFieldWindow < smallestFieldWindow) {
          smallestFieldWindow = tempSmallestFieldWindow;
        }
      }
      break;
    case "anchor":
      if (d.anchors == null) {
        return Integer.MAX_VALUE;
      }
      for (String anchor : d.anchors.keySet()) {
        tempSmallestFieldWindow = getSmallestFieldWindow(tfQuery, anchor, " ");
        if (tempSmallestFieldWindow < smallestFieldWindow) {
          smallestFieldWindow = tempSmallestFieldWindow;
        }
      }
      break;
    }
    return smallestFieldWindow;
  }

  /**
   * get smallest window of one document and query pair.
   * @param d: document
   * @param q: query
   */
  public int getWindow(Document d, Query q, Map<String,Double> tfQuery) {
    int smallestWindow = Integer.MAX_VALUE;
    for (String tfType : this.TFTYPES) {
      int smallestFieldWindow = getSmallestFieldWindow(tfQuery, d, tfType);
      if (smallestFieldWindow < smallestWindow) {
        smallestWindow = smallestFieldWindow;
      }
    }
    return smallestWindow;
  }

  /**
   * get boost score of one document and query pair.
   * @param d: document
   * @param q: query
   */
  private double getSmallestWindowScore (Document d, Query q, Map<String,Double> tfQuery) {
    int smallestWindow = getWindow(d, q, tfQuery);
    double queryLength = 0;
    for (String query : tfQuery.keySet()) {
      queryLength = queryLength + tfQuery.get(query);
    }
    if (smallestWindow == Integer.MAX_VALUE) {
      return 1;
    }
    double diff = smallestWindow - queryLength;
    return 1 + (Bo - 1) * Math.exp(-diff);
  }

  public double[] extractFeatureVector(Document d, Query q){
    
    /* Compute doc_vec and query_vec */
    Map<String,Map<String, Double>> tfs = Util.getDocTermFreqs(d,q);  
    Map<String,Double> queryVector = getQueryVec(q);

    // normalize term-frequency
    this.normalizeTFs(tfs, d, q);
    
    /* [url, title, body, header, anchor] */
    if (Config.isRazarEnabledPart3){
      double[] result = new double[4];
      for (int i = 0; i < result.length; i++) { result[i] = 0.0; }
      for (String queryWord : q.queryWords){
        double queryScore = queryVector.get(queryWord);
        result[0]  += tfs.get("url").get(queryWord) * queryScore;
        result[1]  += tfs.get("body").get(queryWord) * queryScore;
        result[2]  += tfs.get("header").get(queryWord) * queryScore;
        result[3]  += tfs.get("anchor").get(queryWord) * queryScore;
      }
      return result;
    }
    double[] result = new double[5];
    for (int i = 0; i < result.length; i++) { result[i] = 0.0; }
    for (String queryWord : q.queryWords){
      double queryScore = queryVector.get(queryWord);
      result[0]  += tfs.get("url").get(queryWord) * queryScore;
      result[1]  += tfs.get("title").get(queryWord) * queryScore;
      result[2]  += tfs.get("body").get(queryWord) * queryScore;
      result[3]  += tfs.get("header").get(queryWord) * queryScore;
      result[4]  += tfs.get("anchor").get(queryWord) * queryScore;
    }

    return result;
  }

  public double extractBM25Score(Document d, Query q){
    Map<String,Map<String, Double>> tfs = Util.getDocTermFreqs(d,q);
    this.normalizeBM25TFs(tfs, d, q);
    Map<String,Double> tfQuery = Util.getRawQueryFreqs(q);
    return this.getBm25Score(tfs,q,tfQuery,d);
  }

  public double extractSmallestWindowScore(Document d, Query q) {
    Map<String,Double> tfQuery = Util.getRawQueryFreqs(q);
    return getSmallestWindowScore(d, q, tfQuery);
  }

  /* Generate query vector */
  public Map<String,Double> getQueryVec(Query q) {
    /* Count word frequency within the query, in most cases should be 1 */
    
    Map<String, Double> tfVector = new HashMap<String, Double>();
    String[] wordInQuery = q.query.toLowerCase().split(" ");
    for (String word : wordInQuery){
      if (tfVector.containsKey(word))
        tfVector.put(word, tfVector.get(word) + 1);
      else
        tfVector.put(word, 1.0);
    }
    
    /* Sublinear Scaling */
    if(isSublinearScaling){
      for (String word : tfVector.keySet()) {
        tfVector.put(word, 1 + Math.log(tfVector.get(word)));
      }
    }
    
    /* Compute idf vector */
    Map<String,Double> idfVector = new HashMap<String,Double>();
    
    for (String queryWord : q.queryWords) {
      if (this.idfs.containsKey(queryWord))
        idfVector.put(queryWord, this.idfs.get(queryWord));
      else {
        idfVector.put(queryWord, Math.log(98998.0)); /* Laplace smoothing */
      }

    }
    
    /* Do dot-product */
    Map<String, Double> queryVector = new HashMap<String, Double>();
    for (String word : q.queryWords) {
      queryVector.put(word, tfVector.get(word) * idfVector.get(word));
    }
    
    return queryVector;
  }

  public void normalizeTFs(Map<String,Map<String, Double>> tfs,Document d, Query q)
  {
    double normalizationFactor = (double)(d.body_length) + (double)(smoothingBodyLength);

    for (String queryWord : q.queryWords)
      for (String tfType : tfs.keySet())
        tfs.get(tfType).put(queryWord, tfs.get(tfType).get(queryWord)/normalizationFactor);
  }
  
  public void normalizeBM25TFs(Map<String,Map<String, Double>> tfs,Document d, Query q) {
    for (String tfType : tfs.keySet()) {
      Map<String, Double> termFreqHolder = tfs.get(tfType);
      for (String term : termFreqHolder.keySet()) {
        double freqOriginal = termFreqHolder.get(term);
        double avgLength = avgLengths.get(tfType);
        double curLength = lengths.get(d).get(tfType);
        double denominator = (curLength / avgLength - 1) * Bf.get(tfType) + 1;
        termFreqHolder.put(term, freqOriginal / denominator);
      }
    }
  }

  public double[] extractMoreFeatures(Document d, Query q, Map<Query,Map<String, Document>> dataMap) {
    
    double[] basic = extractFeatureVector(d, q);
    double[] more = new double[3];
    double[] result = new double[basic.length+more.length];

    // add page rank as feature
    more[0] = d.page_rank;

    // add bm25 as feature
    more[1] = extractBM25Score(d, q);

    // add smallest window as feature
    more[2] = extractSmallestWindowScore(d, q);

    for (int i=0;i<basic.length;++i){
      result[i]=basic[i];
    }
    for (int i=0;i<more.length;++i){
      result[i+basic.length]=more[i];
    }
    //Additional features et added here:
    /*
     * @TODO: Your code here
     */
    return result;
  }
  
}
