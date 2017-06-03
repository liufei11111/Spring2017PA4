package cs276.pa4;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import java.util.Set;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Implements Pairwise learner that can be used to train SVM
 *
 */
public class PairwiseLearner extends Learner {
  private LibSVM model;
  public PairwiseLearner(boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
  }
  
  public PairwiseLearner(double C, double gamma, boolean isLinearKernel){
    try{
      model = new LibSVM();
    } catch (Exception e){
      e.printStackTrace();
    }
    
    model.setCost(C);
    model.setGamma(gamma); // only matter for RBF kernel
    if(isLinearKernel){
      model.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
    }
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
    linearSVM = Util.loadSignalFileLinearSVMTraining( train_data_file,  idfs, train_rel_file);
    return linearSVM.getFirst();
  }

  @Override
  public Classifier training(Instances dataset) {
    /*
     * @TODO: Your code here
     * Build classifer
     */
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
     * Use this to build the test features that will be used for testing
     */
    Quad<Instances, List<Pair<Query, Pair<Document,Document>>>, ArrayList<Attribute>, Map<Integer, List<Integer>>>
        testData = Util.loadSignalFileLinearSVM(test_data_file,idfs,null);
    TestFeatures testFeas = new TestFeatures();
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
  public Map<Query, List<Document>> testing(TestFeatures tf,
      Classifier model) {
    /*
     * @TODO: Your code here
     */
    Map<Query, List<Document>> result = new HashMap<>();
    Map<Query, List<Document>> grouping = tf.svmQueryDocGrouping;

    for(Query query : grouping.keySet()){
      Map<Pair<Document,Document>, Integer> findPairIndex = tf.svmFeasMap.get(query);
      List<Document> docsPerQuery = grouping.get(query);
      orderDocs(docsPerQuery, findPairIndex, model, tf);
      result.put(query,docsPerQuery);
    }
    return result;
  }

  private void orderDocs(List<Document> docsPerQuery,
      Map<Pair<Document, Document>, Integer> findPairIndex, Classifier model, TestFeatures tf) {
     quickSort(docsPerQuery,0,docsPerQuery.size()-1,findPairIndex, model, tf);
  }


  private void quickSort(List<Document> input,int indexS, int indexE,
      Map<Pair<Document, Document>, Integer> findPairIndex, Classifier model, TestFeatures tf){
    if (indexS < indexE){
      int parition = partition(input, indexS, indexE, findPairIndex, model,tf);
      quickSort(input,indexS,parition-1,findPairIndex,model,tf);
      quickSort(input,parition+1,indexE,findPairIndex,model,tf);
    }
  }

  private int partition(List<Document> input, int indexS, int indexE,
      Map<Pair<Document, Document>, Integer> findPairIndex, Classifier model, TestFeatures tf) {
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
          if (model.classifyInstance(tf.features.get(rawIndex)) > 0.0) {
            i++;    // increment index of smaller element
            swap(input,i,j);
          }
        }else{
//          rawIndex = findPairIndex.get(new Pair(pivot, input.get(j)));
          if (model.classifyInstance(tf.features.get(rawIndex)) == 0.0) {
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
