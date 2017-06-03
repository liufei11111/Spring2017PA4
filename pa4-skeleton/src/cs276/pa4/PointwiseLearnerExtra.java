package cs276.pa4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.util.Map.Entry;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.classifiers.functions.LinearRegression;

/**
 * Implements point-wise learner that can be used to implement logistic regression
 *
 */
public class PointwiseLearnerExtra extends Learner {

  @Override
  public Instances extractTrainFeatures(String train_data_file,
      String train_rel_file, Map<String, Double> idfs) {
//
//    /*
//     * @TODO: Below is a piece of sample code to show
//     * you the basic approach to construct a Instances
//     * object, replace with your implementation.
//     */
//
//    Instances dataset = null;
//
//    /* Build attributes list */
//    ArrayList<Attribute> attributes = new ArrayList<Attribute>();
//    attributes.add(new Attribute("url_w"));
//    attributes.add(new Attribute("title_w"));
//    attributes.add(new Attribute("body_w"));
//    attributes.add(new Attribute("header_w"));
//    attributes.add(new Attribute("anchor_w"));
//    attributes.add(new Attribute("relevance_score"));
//    dataset = new Instances("train_dataset", attributes, 0);
//
//    /* Add data */
//    double[] instance = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
//    Instance inst = new DenseInstance(1.0, instance);
//    dataset.add(inst);
//
//    /* Set last attribute as target */
//    dataset.setClassIndex(dataset.numAttributes() - 1);
//
    Quad<Instances, List<Pair<Query, Document>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
        trainData = UtilExtraSVOReg.loadSignalFileLinearRegression(train_data_file,idfs,train_rel_file);
    return trainData.getFirst();
  }

  @Override
  public Classifier training(Instances dataset) {
    /*
     * @TODO: Your code here
     */
    LinearRegression model = new LinearRegression();
    try {
      model.buildClassifier(dataset);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return model;
  }

  @Override
  public TestFeatures extractTestFeatures(String test_data_file,
      Map<String, Double> idfs) {
    /*
     * @TODO: Your code here
     * Create a TestFeatures object
     * Build attributes list, instantiate an Instances object with the attributes
     * Add data and populate the TestFeatures with the dataset and features
     */
    Quad<Instances, List<Pair<Query, Document>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
        testData = UtilExtraSVOReg.loadSignalFileLinearRegression(test_data_file,idfs,null);
    TestFeatures testFeas = new TestFeatures();
    testFeas.features = testData.getFirst();
    HashMap<Query, Map<Document,Integer>> indexMap = new HashMap<>();
    List<Pair<Query, Document>> rows = testData.getSecond();

    for (int i=0;i< rows.size();++i){
      Pair<Query, Document> pair = rows.get(i);
      if (!indexMap.containsKey(pair.getFirst())){
        indexMap.put(pair.getFirst(),new HashMap<Document, Integer>());
      }
      indexMap.get(pair.getFirst()).put(pair.getSecond(),i);
    }
    testFeas.index_map = indexMap;
    return testFeas;
  }

  @Override
  public Map<Query, List<Document>> testing(TestFeatures tf,
      Classifier model) {
    /*
     * @TODO: Your code here
     */
    Map<Query, List<Document>> result = new HashMap<>();
    List<Pair<Document,Double>> array = new ArrayList<>();
    Map<Query, Map<Document, Integer>> indexMap = tf.index_map;
    for (Entry<Query, Map<Document, Integer>> entry : indexMap.entrySet()){
      for (Entry<Document, Integer> entryDoc : entry.getValue().entrySet()){
        double docScore = 0;
        try {
          docScore = model.classifyInstance(tf.features.get(entryDoc.getValue()));
          array.add(new Pair(entryDoc.getKey(), docScore));
        } catch (Exception e) {
          e.printStackTrace();
        }

      }
      Collections.sort(array, new Comparator<Pair<Document, Double>>() {
        @Override
        public int compare(Pair<Document, Double> o1, Pair<Document, Double> o2) {
          return o2.getSecond().compareTo(o1.getSecond());
        }
      });
      List<Document> sorted = new ArrayList<>();
      for (Pair<Document,Double> scoreEntry : array){
        sorted.add(scoreEntry.getFirst());
      }
      result.put(entry.getKey(),sorted);
      array.clear();
    }
    return result;
  }

}
