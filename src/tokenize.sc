//> using scala "3.6.4"
//> using dep com.softwaremill.sttp.client4::core:4.0.3
//> using dep com.softwaremill.sttp.client4::upickle:4.0.3
//> using dep com.lihaoyi::os-lib:0.11.4
//> using dep com.github.scopt::scopt:4.1.0
//> using dep org.scalameta::scalameta:4.13.5

import upickle.default.*
import scala.meta.*
import scopt.OParser
import os.{Path, RelPath, pwd}
import java.time.Instant

object Logger:
  val runTimestamp = Instant.now().toEpochMilli
  val logFile = pwd / s"tokenizer_log_${runTimestamp}.txt"

  def info(message: String): Unit =
    os.write.append(logFile, s"[INFO] $message\n")
    println(s"[INFO] $message")

  def error(message: String): Unit =
    os.write.append(logFile, s"[ERROR] $message\n")
    println(s"[ERROR] $message")

case class CliArgs(
  inputFile: String = "",
  outputDir: String = "",
  minFrequency: Int = 1,
  maxVocabSize: Option[Int] = None
)

case class CodeSnippet(
  id: String,
  timestamp: Long,
  code: String,
  category: String,
  topicUsed: String,
  isIdiomatic: Boolean
) derives ReadWriter

case class CodeSnippetWithTokens(
  id: String,
  timestamp: Long,
  code: String,
  category: String,
  topicUsed: String,
  isIdiomatic: Boolean,
  tokens: Vector[String]
) derives ReadWriter

def tokenize(tree: Tree): Vector[String] =
  val tokens = collection.mutable.ArrayBuffer[String]()

  inline def splitLexeme(s: String): List[String] =
    """[A-Z]?[a-z]+|[0-9]+|[A-Z]+(?=[A-Z]|$)""".r
      .findAllMatchIn(s)
      .map(_.matched.toLowerCase)
      .toList

  def dfs(node: Tree): Unit =
    tokens += node.productPrefix

    // node match
    //   case Lit.Int(value)    => tokens ++= List(value.toString)
    //   case Lit.String(value) => tokens ++= List(value.toString)
    //   case Lit.Long(value)   => tokens ++= List(value.toString)
    //   case Lit.Double(value) => tokens ++= List(value.toString)
    //   case name: Name        => tokens ++= List(name.value)
    //   case _                 => ()

    node match
      case Lit.Int(value)    => tokens ++= splitLexeme(value.toString)
      case Lit.String(value) => tokens ++= splitLexeme(value.toString)
      case Lit.Long(value)   => tokens ++= splitLexeme(value.toString)
      case Lit.Double(value) => tokens ++= splitLexeme(value.toString)
      case name: Name        => tokens ++= splitLexeme(name.value)
      case _                 => ()

    val children = node.children
    if children.nonEmpty then
      tokens += "<Down>"
      children.foreach(dfs)
      tokens += "<Up>"

  dfs(tree)
  tokens.toVector

def buildVocab(
  corpus: Seq[Seq[String]],
  specials: Seq[String] = Seq("<PAD>", "<UNK>"),
  minFreq: Int = 1,
  maxSize: Option[Int] = None
): Vector[String] =
  val freq = corpus.flatten.groupMapReduce(identity)(_ => 1)(_ + _)
  val sorted = freq.toVector
    .filter((_, f) => f >= minFreq)
    .sortBy((tok, f) => (-f, tok)) // frequency desc, then alpha
  val limited = maxSize.fold(sorted)(n => sorted.take(n))
  (specials ++ limited.map(_._1)).toVector

def parseCommandLine(args: Array[String]): Option[CliArgs] = {
  val builder = OParser.builder[CliArgs]
  import builder.*

  val parser = OParser.sequence(
    programName("scala-ast-tokenizer"),
    head("Scala AST Tokenizer", "1.0"),
    opt[String]("input")
      .required()
      .action((path, config) => config.copy(inputFile = path))
      .text("Input file path with code snippets in JSONL format"),
    opt[String]("output-dir")
      .required()
      .action((path, config) => config.copy(outputDir = path))
      .text("Output directory path"),
    opt[Int]("min-freq")
      .action((n, config) => config.copy(minFrequency = n))
      .text("Minimum frequency for tokens to include in vocabulary"),
    opt[Int]("max-vocab")
      .action((n, config) => config.copy(maxVocabSize = Some(n)))
      .text("Maximum vocabulary size")
  )

  OParser.parse(parser, args, CliArgs())
}

def processSnippets(
  inputPath: Path,
  minFreq: Int,
  maxSize: Option[Int]
): (Seq[CodeSnippetWithTokens], Vector[String]) = {
  Logger.info(s"Processing snippets from $inputPath")

  val lines = os.read.lines(inputPath)
  Logger.info(s"Loaded ${lines.size} snippets")

  val processedSnippets = lines.flatMap { line =>
    val snippet = read[CodeSnippet](line)
    val codeStr = snippet.code

    def tryParse[T <: Tree](parsed: => Parsed[T]): Option[T] =
      try parsed.toOption
      catch case _: Throwable => None

    val ast: Option[Tree] =
      tryParse(codeStr.parse[Source])
        .orElse(tryParse(dialects.Scala3.withAllowToplevelTerms(true).apply(codeStr).parse[Source]))
        .orElse(tryParse(dialects.Scala3.withAllowToplevelTerms(true).apply(codeStr).parse[Stat]))
        .orElse(tryParse(dialects.Scala3.withAllowToplevelTerms(true).apply(codeStr).parse[Type]))

    ast match
      case Some(tree) =>
        val tokens = tokenize(tree)
        Some(
          CodeSnippetWithTokens(
            id = snippet.id,
            timestamp = snippet.timestamp,
            code = snippet.code,
            category = snippet.category,
            topicUsed = snippet.topicUsed,
            isIdiomatic = snippet.isIdiomatic,
            tokens = tokens
          )
        )
      case None =>
        Logger.error(s"Failed to parse snippet ${snippet.id}:\n$codeStr")
        None
  }.toSeq

  Logger.info(s"Successfully tokenized ${processedSnippets.size} out of ${lines.size} snippets")

  val corpus = processedSnippets.map(_.tokens)
  val vocab = buildVocab(corpus, minFreq = minFreq, maxSize = maxSize)
  Logger.info(s"Built vocabulary with ${vocab.size} tokens")

  (processedSnippets, vocab)
}

parseCommandLine(args) match {
  case Some(cliArgs) =>
    val inputPath = pwd / RelPath(cliArgs.inputFile)
    val outPath = pwd / RelPath(cliArgs.outputDir)

    os.makeDir.all(outPath)

    if (!os.exists(inputPath)) {
      Logger.error(s"Input file does not exist: $inputPath")
      System.exit(1)
    }

    try {
      val (processedSnippets, vocab) = processSnippets(inputPath, cliArgs.minFrequency, cliArgs.maxVocabSize)
      val tokensFile = outPath / "processed_full.jsonl"
      val vocabFile = outPath / "vocab.json"

      val outputLines = processedSnippets.map(snippet => write(snippet))
      os.write.over(tokensFile, outputLines.mkString("\n"))
      Logger.info(s"Saved ${processedSnippets.size} tokenized snippets to $tokensFile")

      os.write.over(vocabFile, write(vocab))
      Logger.info(s"Saved vocabulary with ${vocab.size} tokens to $vocabFile")

      Logger.info("=== Processing complete ===")
      Logger.info(s"Total snippets processed: ${processedSnippets.size}")
      Logger.info(s"Vocabulary size: ${vocab.size}")
    } catch {
      case ex: Exception =>
        Logger.error(s"Error processing snippets: ${ex.getMessage}")
        ex.printStackTrace()
        System.exit(1)
    }

  case None =>
    System.exit(1)
}
