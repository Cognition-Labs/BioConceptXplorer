import React, { useState, useEffect } from "react";
import Autosuggest from "react-autosuggest";
import { StyleSheet, css } from "aphrodite";
import { useRecoilState } from "recoil";
import { expressionAtom } from "../atoms";
import { Box } from "@chakra-ui/react";

const autosuggestStyles = StyleSheet.create({
  container: {
    position: "relative",
  },
  suggestionsContainer: {
    position: "absolute",
    zIndex: 1,
    marginTop: "2px",
    backgroundColor: "#fff",
    width: "100%",
    border: "1px solid #d9d9d9",
    borderRadius: "4px",
  },
  suggestion: {
    padding: "10px 20px",
    cursor: "pointer",
  },
  suggestionHighlighted: {
    backgroundColor: "#f0f0f0", // Customize the background color for the highlighted suggestion
  },
  input: {
    width: "100%",
    height: "30px",
    padding: "0.375rem 0.75rem",
    borderWidth: "1px",
    borderRadius: "0.25rem",
  },
});

// Apply the styles using css() from aphrodite
const styledAutosuggestStyles = {
  container: css(autosuggestStyles.container),
  suggestionsContainer: css(autosuggestStyles.suggestionsContainer),
  suggestion: css(autosuggestStyles.suggestion),
  input: css(autosuggestStyles.input),
};

const ExploreCreator: React.FC = () => {
  // State variables
  const [expression, setExpression] = useRecoilState(expressionAtom);
  const [input, setInput] = useState("");
  const [selectedSide, setSelectedSide] = useState<"left" | "right">("left");
  const [isFocused, setIsFocused] = useState(false);
  const [selectedToken, setSelectedToken] = useState<{
    side: "left" | "right";
    index: number;
  } | null>(null);
  const [suggestions, setSuggestions] = useState<string[][]>([]);


  useEffect(() => {
    let suggestions_: string[][] = []
    let expression_ = expression.filter((token) => token != "+" && token != "-");
    expression_.forEach((token, idx) => {
      // make api call for top k and save to state
      fetch("https://shreyj1729--bioconceptvec-get-similar-concepts.modal.run?concept_query=" + token + "&k=4", {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      }).then(res => res.json()).then(data => {
        console.log(data)
        suggestions.push(data)
      }).catch(err => {
        console.log(err)
      })
    })
    setSuggestions(suggestions_)
  }, [expression])

  // Function to get suggestion value
  // const getSuggestionValue = (suggestion: string) => suggestion;

  // Function to render suggestion
  const renderSuggestion = (suggestion: string, { isHighlighted }: any) => (
    <div
      className={
        isHighlighted
          ? css(autosuggestStyles.suggestionHighlighted)
          : css(autosuggestStyles.suggestion)
      }
    >
      {suggestion}
    </div>
  );

  // Function to handle input change
  const onChange = (event: any, { newValue }: any) => {
    setInput(newValue);
  };

  // Function to handle suggestions clear requested
  const onSuggestionsClearRequested = () => {
    setSuggestions([]);
  };

  // Autosuggest input properties
  const inputProps = {
    placeholder: "",
    value: input,
    onChange: onChange,
  };

  return (
    <>
      <Box mt={5}>
        {expression.map((token, i) => {
          return (
            <Box
              p={2}
              border="1px"
              key={i}
              as="span"
              color={selectedToken?.index === i ? "blue.900" : "whiteAlpha.900"}
              bg={selectedToken?.index === i ? "blue.100" : "blackAlpha.700"}
              // if selected token, hand on hover
              cursor={selectedToken?.index === i ? "pointer" : "default"}
              onClick={() => {
                setSelectedToken({ side: selectedSide, index: i });
              }}
            >
              {token}
            </Box>
          );
        })}
      </Box>

      <Box>
        {/* map over equation terms, for each show list of similar terms */}
        {expression.map((token, i) => {
          return (
            <>
              {suggestions[i] && suggestions[i].length > 0 && (
                <Box>
                  {suggestions[i].map((suggestion: string, j: number) => {
                    return (
                      <Box
                        p={2}
                        border="1px"
                        key={j}
                        as="span"
                        color={selectedToken?.index === i ? "blue.900" : "whiteAlpha.900"}
                        bg={selectedToken?.index === i ? "blue.100" : "blackAlpha.700"}
                        // if selected token, hand on hover
                        cursor={selectedToken?.index === i ? "pointer" : "default"}
                        onClick={() => {
                          setSelectedToken({ side: selectedSide, index: i });
                        }}
                      >
                        {suggestion}
                      </Box>
                    );

                  })}
                </Box>
              )
              }
            </>
          )
        })}
      </Box >
    </>
  );
};

export default ExploreCreator;
