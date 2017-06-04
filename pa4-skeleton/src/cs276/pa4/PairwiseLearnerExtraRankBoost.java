package cs276.pa4;

import ciir.umass.edu.eval.Evaluator;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.boosting.RankBoost;
import cs276.pa4.ranklib.RankLibWrapper;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Created by feiliu on 5/29/17.
 */
public class PairwiseLearnerExtraRankBoost extends Learner{
  private RankBoost model;
  public PairwiseLearnerExtraRankBoost(){
    model = new RankBoost();

  }



  @Override
  public Instances extractTrainFeatures(String train_data_file,
      String train_rel_file, Map<String, Double> idfs) {
    /*
     * @TODO: Your code here:
     * Get signal file
     * Construct output dataset of type Instances
     * Add new attribute  to store relevance in the train dataset
     * Populate data
     */
    Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
        linearSVM = UtilExtraRankLib.loadSignalFileLinearSVMTraining( train_data_file,  idfs, train_rel_file);
    return linearSVM.getFirst();
  }

  @Override
  public Classifier training(Instances dataset) {
    trainingRankBoost();
    return null;
  }

  public RankBoost trainingRankBoost() {
    System.out.println("Starting the training of model");
    Evaluator.main(new String[]{
        "-train", Config.saveRankNetTrainFileName,
        "-metric2t", "NDCG@10",
        "-ranker", "2",
        "-frate", "1.0",
        "-bag", "10",
        "-round", "10",
        "-epoch", "10",
        "-save", Config.saveRankNetModelFileName});
    System.out.println("Ending the training of model");
    return null;
  }

  @Override
  public TestFeatures extractTestFeatures(String test_data_file,
      Map<String, Double> idfs) {
    /*
     * @TODO: Your code here
     * Use this to build the test features that will be used for testing
     */
    TestFeatures testFeas = new TestFeatures();
    Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
        testData = UtilExtraRankLib.loadSignalFileLinearSVM(test_data_file,idfs,null,testFeas);

    testFeas.features = testData.getFirst();
    HashMap<Query, Map<Pair<Document,Document>,Integer>> indexMap = new HashMap<>();
    List<Pair<Query, Pair<Document,Document>>> rows = testData.getSecond();
    Map<Query,List<Document>> grouping = new HashMap<>();
    Set<Document> duplicateCheck = new HashSet<>();
    Query prev = null;
    for (int i=0;i < rows.size();++i){
      Pair<Query, Pair<Document,Document>> pair = rows.get(i);
      if (prev == null ||  prev != pair.getFirst()) {
        grouping.put(pair.getFirst(),new ArrayList<Document>());
        duplicateCheck.clear();
      }
      if (!duplicateCheck.contains(pair.getSecond().getFirst())){
        grouping.get(pair.getFirst()).add(pair.getSecond().getFirst());
        duplicateCheck.add(pair.getSecond().getFirst());
      }

      prev = pair.getFirst();
      //
      if (!indexMap.containsKey(pair.getFirst())){
        indexMap.put(pair.getFirst(),new HashMap<Pair<Document,Document>,Integer>());
      }
      indexMap.get(pair.getFirst()).put(pair.getSecond(),i);
    }
    testFeas.svmFeasMap = indexMap;
    testFeas.svmQueryDocGrouping = grouping;
    return testFeas;
  }

  @Override
  public Map<Query, List<Document>> testing(TestFeatures tf, Classifier model) {
    System.out.println("Starting the testing of model");
    RankBoost rb = RankLibWrapper.loadModel();
    System.out.println("Loaded the RankBoost model");
    return testing( tf, rb);
  }


  public Map<Query, List<Document>> testing(TestFeatures tf,
      RankBoost model) {
    /*
     * @TODO: Your code here
     */
    Map<Query, List<Document>> result = new HashMap<>();
    Map<Query, List<Document>> grouping = tf.svmQueryDocGrouping;

    for(Query query : grouping.keySet()){
      Map<Pair<Document,Document>, Integer> findPairIndex = tf.svmFeasMap.get(query);
      List<Document> docsPerQuery = grouping.get(query);
      orderDocs(docsPerQuery, findPairIndex, model, tf,tf.rowToQueryID);
      result.put(query,docsPerQuery);
    }
    System.out.println("Done testing the model");
    return result;
  }

  private void orderDocs(List<Document> docsPerQuery,
      Map<Pair<Document, Document>, Integer> findPairIndex, RankBoost model, TestFeatures tf,Map<Integer, Integer> rowToQueryID) {
    quickSort(docsPerQuery,0,docsPerQuery.size()-1,findPairIndex, model, tf,rowToQueryID);
  }


  private void quickSort(List<Document> input,int indexS, int indexE,
      Map<Pair<Document, Document>, Integer> findPairIndex, RankBoost model, TestFeatures tf,Map<Integer, Integer> rowToQueryID){
    if (indexS < indexE){
      int parition = partition(input, indexS, indexE, findPairIndex, model,tf,rowToQueryID);
      quickSort(input,indexS,parition-1,findPairIndex,model,tf,rowToQueryID);
      quickSort(input,parition+1,indexE,findPairIndex,model,tf,rowToQueryID);
    }
  }

  private int partition(List<Document> input, int indexS, int indexE,
      Map<Pair<Document, Document>, Integer> findPairIndex, RankBoost model, TestFeatures tf,Map<Integer, Integer> rowToQueryID) {
//    System.out.println("IndexS: "+indexS +", IndexE: "+indexE);
//    System.out.println("Documents");
//    for (Document doc : input){
//      System.out.print(doc + " ,");
//    }
    Document pivot = input.get(indexE);
    int i = (indexS - 1); // Index of smaller element

    for (int j = indexS; j <= indexE- 1; j++) {
      // If current element is smaller than or
      // equal to pivot
      try {
        Integer rawIndex = findPairIndex.get(new Pair(input.get(j), pivot));
        if (rawIndex == null){
          rawIndex = findPairIndex.get(new Pair(pivot, input.get(j)));
          if (rawIndex == null){
            throw new RuntimeException("An order can not be established $$$###");
          }
          if (model.eval(RankLibWrapper.convertToDataPoint(rawIndex, tf.features.get(rawIndex),rowToQueryID)) > 0.0) {
            i++;    // increment index of smaller element
            swap(input,i,j);
          }
        }else{
//          rawIndex = findPairIndex.get(new Pair(pivot, input.get(j)));
          if (model.eval(RankLibWrapper.convertToDataPoint(rawIndex, tf.features.get(rawIndex),rowToQueryID)) == 0.0) {
            i++;    // increment index of smaller element
            swap(input,i,j);
          }
        }


      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    swap(input,i+1,indexE);
    return (i + 1);
  }

  private void swap(List<Document> input, int i, int j){
    Document temp = input.get(i);
    input.set(i,input.get(j));
    input.set(j,temp);
  }
}
