import { AfterViewInit, Component, ElementRef, ViewChild } from "@angular/core";

@Component({
    selector: "app-canvas",
    templateUrl: "canvas.component.html",
    styleUrls: ["canvas.component.css"],
})
export class CanvasComponent implements AfterViewInit {
    @ViewChild("scrollable")
    private _scrollableRef!: ElementRef<HTMLDivElement>;

    ngAfterViewInit(): void {
        const { nativeElement } = this._scrollableRef;
        const { scrollHeight, clientHeight, scrollWidth, clientWidth } =
            nativeElement;

        nativeElement.scrollTo({
            top: (scrollHeight - clientHeight) / 2,
            left: (scrollWidth - clientWidth) / 2,
            behavior: "instant",
        });
    }
}
