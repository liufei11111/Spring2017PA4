package cs276.pa4;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Main entry-point of PA4
 * Version 2.0: includes idfs_file as a command line argument
 */
public class Learning2RankPart2 {

  /**
   * Returns a trained model
   * @param train_signal_file
   * @param train_rel_file
   * @param task
   *      1: Linear Regression
   *      2: SVM
   *      3: More features
   *      4: Extra credit
   * @param idfs
   * @return
   */
  public static Classifier train(String train_signal_file, String train_rel_file, int task, Map<String,Double> idfs) {
    System.err.println("## Training with feature_file =" + train_signal_file + ", rel_file = " + train_rel_file + " ... \n");
    Classifier model = null;
    Learner learner = null;

    if (task == 1) {
      learner = new PointwiseLearner();
    } else if (task == 2) {
      boolean isLinearKernel = true;
//       learner = new PairwiseLearner(isLinearKernel);
      learner= new PairwiseLearner(Config.C, Config.gamma, false);
    } else if (task == 3) {

      /*
       * @TODO: Your code here, add more features
       * */
      System.err.println("Task 3");
            /*
       * @TODO: Your code here, add more features
       * */
      System.err.println("Task 3");
      learner = new PairwiseLearnerPart3(Config.C, Config.gamma, false);
    } else if (task == 4) {

      /*
       * @TODO: Your code here, extra credit
       * */

      System.err.println("Extra credit");
      learner = new PairwiseLearnerExtra(Config.C, Config.gamma, false);
    }

    /* Step (1): construct your feature matrix here */
    Instances data = learner.extractTrainFeatures(train_signal_file, train_rel_file, idfs);

    /* Step (2): implement your learning algorithm here */
    model = learner.training(data);

    return model;
  }

  /**
   * Test model using the test signal file
   * @param test_signal_file
   * @param model
   * @param task
   * @param idfs
   * @return
   */
  public static Map<Query, List<Document>> test(String test_signal_file, Classifier model, int task, Map<String,Double> idfs){
    System.err.println("## Testing with feature_file=" + test_signal_file + " ... \n");
    Map<Query, List<Document>> ranked_queries = new HashMap<Query, List<Document>>();
    Learner learner = null;
    if (task == 1) {
      learner = new PointwiseLearner();
    } else if (task == 2) {
      boolean isLinearKernel = true;
//      learner = new PairwiseLearner(isLinearKernel);
      learner= new PairwiseLearner(Config.C, Config.gamma, false);
    } else if (task == 3) {

      /*
       * @TODO: Your code here, add more features
       * */
      System.err.println("Task 3");
      learner = new PairwiseLearnerPart3(Config.C, Config.gamma, false);

    } else if (task == 4) {

      /*
       * @TODO: Your code here, extra credit
       * */
      System.err.println("Extra credit");
      learner = new PairwiseLearnerExtra(Config.C, Config.gamma, false);
    }
    /* Step (1): construct your test feature matrix here */
    TestFeatures tf = learner.extractTestFeatures(test_signal_file, idfs);

    /* Step (2): implement your prediction and ranking code here */
    ranked_queries = learner.testing(tf, model);

    return ranked_queries;
  }


  /**
   * Output the ranking results in expected format
   * @param
   * @param ps
   */
  public static void writeRankedResultsToFile(Map<Query,List<Document>> queryRankings,PrintStream ps) {
    for (Query query : queryRankings.keySet()) {
      StringBuilder queryBuilder = new StringBuilder();
      for (String s : query.queryWords) {
        queryBuilder.append(s);
        queryBuilder.append(" ");
      }

      String queryStr = "query: " + queryBuilder.toString() + "\n";
      ps.print(queryStr);
//      System.out.println("query: " + query);
//      System.out.println("queryRankings.get(query): "+queryRankings.get(query).size());
      for (Document res : queryRankings.get(query)) {
        String urlString =
            "  url: " + res.url + "\n" +
                "    title: " + res.title + "\n" +
                "    debug: " + res.debugStr + "\n";
        ps.print(urlString);
      }
    }
  }

  public static void main(String[] args) throws IOException {
    if (args.length != 5 && args.length != 6) {
      System.err.println("Input arguments: " + Arrays.toString(args));
      System.err.println("Usage: <train_signal_file> <train_rel_file> <test_signal_file> <idfs_file> <task> [ranked_out_file]");
      System.err.println("  ranked_out_file (optional): output results are written into the specified file. If not, output to stdout.");
      return;
    }
    String train_signal_file = args[0];
    String train_rel_file = args[1];
    String test_signal_file = args[2];
    String dfFile = args[3];
    int task = Integer.parseInt(args[4]);
    String ranked_out_file = "";
    if (args.length == 6){
      ranked_out_file = args[5];
    }

    /* Populate idfs */
    Map<String,Double> idfs = Util.loadDFs(dfFile);

    /* Train & test */
    System.err.println("### Running task" + task + "...");
    double highestScore = -Double.MAX_VALUE;
    double highestC = -Double.MAX_VALUE;
    double highestGamma = -Double.MAX_VALUE;
    double total = 10.0;
    for (double i=0;i<total;i=i+1.0){
//      Config.C = 56.0+i*2.0/total;
//      Config.C = Math.pow(2,-3+i);
//      for(double j=0;j<total;j=j+1.0){
        Config.gamma = 0.015+0.04*i/total;
        Classifier model = train(train_signal_file, train_rel_file, task, idfs);
    /* performance on the training data */
        Map<Query, List<Document>> trained_ranked_queries = test(train_signal_file, model, task, idfs);
        String trainOutFile="tmp.train.ranked";
        writeRankedResultsToFile(trained_ranked_queries, new PrintStream(new FileOutputStream(trainOutFile)));
        NdcgMain ndcg = new NdcgMain(train_rel_file);
        double tempScore =ndcg.score(trainOutFile);
        if (highestScore < tempScore ){
          highestScore=tempScore;
          highestC=Config.C;
          highestGamma=Config.gamma;
        }
        System.err.println("# Trained NDCG=" + tempScore);
        (new File(trainOutFile)).delete();

        Map<Query, List<Document>> ranked_queries = test(test_signal_file, model, task, idfs);

    /* Output results */
        if(ranked_out_file == null || ranked_out_file.isEmpty()){ /* output to stdout */
          writeRankedResultsToFile(ranked_queries, System.out);
        } else {
      /* output to file */
          try {
            writeRankedResultsToFile(ranked_queries, new PrintStream(new FileOutputStream(ranked_out_file)));
          } catch (FileNotFoundException e) {
            e.printStackTrace();
          }
        }
//      }

    }
    System.out.println("Highest C: "+highestC);
    System.out.println("HighestGamma: "+highestGamma);
    System.out.println("NDCG: "+highestScore);

  }
}

