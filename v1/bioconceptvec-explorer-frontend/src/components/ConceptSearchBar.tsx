import { Flex, FormControl, FormHelperText, FormLabel, useColorModeValue } from "@chakra-ui/react";
import * as React from "react"
import {
    AutoComplete,
    AutoCompleteInput,
    AutoCompleteItem,
    AutoCompleteList,
} from "@choc-ui/chakra-autocomplete";
import { Item } from "@choc-ui/chakra-autocomplete";
import { expressionAtom, loadingState } from "../atoms";
import { useRecoilState } from "recoil";

export const ConceptSearchBar = () => {
    const [expression, setExpression] = useRecoilState(expressionAtom)

    const addConceptToEq = (params: { item: Item, selectMethod: "mouse" | "keyboard" | null, isNewInput?: boolean }) => {
        console.log(params.item.value)
        if (expression.length === 1 && expression[0] === "") {
            setExpression([params.item.value])
        } else if (expression[-1] !== "+") {
            setExpression([...expression, "+", params.item.value]);
        } else {
            setExpression([...expression, params.item.value]);
        }
    }

    const [searchLoading, setSearchLoading] = useRecoilState(loadingState)

    const queryBackend = (event: React.ChangeEvent<HTMLInputElement>) => {
        let query = event.target.value;
        if (query.length > 0) {
            console.log(query)
            setSearchLoading(true)
            fetch(`https://shreyj1729--bioconceptvec-autosuggest.modal.run/?query=${query}&limit=100`)
                .then(response => response.json())
                .then(data => {
                    setConcepts(data)
                    setSearchLoading(false)
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
        }
    }

    const [concepts, setConcepts] = React.useState(["type to see results..."])

    const [searchbarBlurred, setSearchbarBlurred] = React.useState(false);

    const textColor = useColorModeValue("inherit", "whiteAlpha.900");

    return (
        <Flex justify="center" w="full">
            <FormControl w="60">
                <FormLabel>Search Concepts</FormLabel>
                <AutoComplete defaultIsOpen openOnFocus onSelectOption={addConceptToEq} suggestWhenEmpty closeOnBlur={false} onBlur={() => { setSearchbarBlurred(true) }} onFocus={() => { setSearchbarBlurred(false) }} disableFilter={searchbarBlurred} maxSuggestions={100} restoreOnBlurIfEmpty={false}>
                    <AutoCompleteInput variant="filled" onChange={queryBackend} />
                    <AutoCompleteList color={textColor} bgColor={useColorModeValue("white", "gray.800")} fontWeight={"normal"}>
                        {concepts.map((concept, cid) => (
                            <AutoCompleteItem
                                key={`option-${cid}`}
                                value={concept}
                                textTransform="capitalize"
                            >
                                {concept}
                            </AutoCompleteItem>
                        ))}
                    </AutoCompleteList>
                </AutoComplete>
            </FormControl>
        </Flex >
    );
}