import { AfterViewInit, Component, ElementRef, ViewChild } from "@angular/core";

@Component({
    selector: "app-canvas",
    templateUrl: "canvas.component.html",
    styleUrl: "canvas.component.css",
})
export class CanvasComponent implements AfterViewInit {
    @ViewChild("scrollable")
    public scrollable!: ElementRef<HTMLDivElement>;

    public ngAfterViewInit(): void {
        const { nativeElement } = this.scrollable;
        const { scrollHeight, clientHeight, scrollWidth, clientWidth } =
            nativeElement;

        nativeElement.scrollTo({
            top: (scrollHeight - clientHeight) / 2,
            left: (scrollWidth - clientWidth) / 2,
            behavior: "instant",
        });
    }
}
