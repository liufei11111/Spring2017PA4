package cs276.pa4;

/**
 * Created by feiliu on 5/29/17.
 */
public class Config {
  public static double part2C = 57.5;
  public static double part2Gamma = 0.031;

  public static double part3C = 30;
  public static double part3Gamma = 0.04675;

  public static String savedEmbeddingFile="/farmshare/user_data/fliu5/savedEmbedding.txt";
//  public static String savedEmbeddingFile="savedEmbedding.txt";
  public static int embeddingSize=300;

  public static double part4C=0.0828125;
  public static double part4Gamma=  0.698;

  // for tunning only
  public static double C = 0.0828125;
  public static double gamma = 0.698;

  // SMO
  public static double part4SMOC=100;
  public static int polyOrder = 4;
  public static double part4SMOL=0.00001;
  // enable razer part 3

  public static boolean isRazarEnabledPart3=true;
  // enable razar part 4
  public static boolean isRazarEnabledPart4Title =false;
  public static boolean isRazarEnabledPart4TitleSim =false;
  // ranknet train feature name
  public static String saveRankNetTrainFileName ="ranknetTrainFeatureName.txt";
  public static String saveRankNetTestFileName="ranknetTestFeatureName.txt";
  public static String saveRankNetModelFileName="ranknetModelFile.txt";
  public static String adaRankTolerance="0.00001";
  public static String adaRankRounds="50";
}
